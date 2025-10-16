import sys
from pathlib import Path
import types

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------- Dummies -------------------------------------

class DummyTestDataset(Dataset):
    """
    Minimal dataset for predict: returns (image, meta) where meta contains a filename.
    """
    def __init__(self, mode, image_paths, image_size=None):
        assert mode == "test"
        self.paths = list(image_paths)
        self.H = self.W = 8 if image_size is None else min(8, int(image_size))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = torch.zeros(3, self.H, self.W, dtype=torch.float32)
        filename = Path(self.paths[idx]).name
        meta = {"filename": filename}
        return x, meta


class DummyEncoder(nn.Module):
    def __init__(self, name, pretrained):
        super().__init__()
        self.d = nn.Parameter(torch.zeros(1))
        # mimic what predict.py expects to pass into the decoder
        self.feature_channels = [64, 128, 256, 512]


class DummyDecoder(nn.Module):
    def __init__(self, encoder_channels=None):
        super().__init__()
        self.d = nn.Parameter(torch.zeros(1))


class DummySegmentationModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # not used by Predictor in the mocked setup
        return torch.zeros(x.size(0), 1, x.size(2), x.size(3))


class SpyPredictor:
    """
    Capture ctor args; return a fixed mask in predict_batch.
    """
    last_init = None

    def __init__(self, model, ckpt_path, device, threshold):
        type(self).last_init = {
            "model": model,
            "ckpt_path": ckpt_path,
            "device": device,
            "threshold": threshold,
        }

    def predict_batch(self, imgs: torch.Tensor) -> torch.Tensor:
        # imgs: (B,3,H,W) -> return (B,1,H,W)
        B, _, H, W = imgs.shape
        # Make a non-trivial mask so we could (optionally) check thresholding logic elsewhere.
        return torch.ones(B, 1, H, W, dtype=torch.float32)


# ----------------------------- Fixtures ------------------------------------

@pytest.fixture
def wire_pred_mocks(monkeypatch, tmp_path):
    """
    Monkeypatch predict.py dependencies with dummies and stub data discovery.
    """
    import dida_roofseg.predict as P

    # 1) Stub discover_pairs to provide only test images (no masks needed here)
    test_imgs = [tmp_path / f"test_{i}.png" for i in range(3)]
    labeled = []   # not used
    mask_map = {}  # not used
    monkeypatch.setattr(
        "dida_roofseg.predict.discover_pairs",
        lambda data_dir: (labeled, mask_map, test_imgs),
        raising=True,
    )

    # 2) Use tiny dataset that yields (img, meta) where meta["filename"] is present
    monkeypatch.setattr("dida_roofseg.predict.RoofDataset", DummyTestDataset, raising=True)

    # 3) Swap out models
    dummy_model_mod = types.SimpleNamespace(
        EncoderWrapper=DummyEncoder,
        DecoderUNetSmall=DummyDecoder,
        SegmentationModel=DummySegmentationModel,
    )
    monkeypatch.setattr("dida_roofseg.predict.model_mod", dummy_model_mod, raising=True)

    # 4) Swap out Predictor
    monkeypatch.setattr("dida_roofseg.predict.Predictor", SpyPredictor, raising=True)

    # 5) Replace save_mask to actually write a small file so we can assert on outputs
    def fake_save_mask(mask: torch.Tensor, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # persist something that proves we were called with the right filename
        Path(out_path).write_text(f"shape={tuple(mask.shape)}")

    monkeypatch.setattr("dida_roofseg.predict.save_mask", fake_save_mask, raising=True)

    return P  # patched module under test


# ------------------------------- Tests -------------------------------------

def test_parse_args_defaults_predict(monkeypatch):
    from dida_roofseg.predict import parse_args
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = parse_args()
    assert args.data_dir == "data/raw"
    assert args.ckpt_path == "models/checkpoints/best.pth"
    assert args.pred_dir == "outputs/predictions"
    assert args.image_size == 512
    assert args.encoder == "resnet18"
    assert args.threshold == 0.5
    assert args.seed == 42
    assert args.device == "cpu"


def test_predict_main_writes_files_and_prints(wire_pred_mocks, monkeypatch, capsys, tmp_path):
    pred_dir = tmp_path / "preds"
    ckpt_path = tmp_path / "best.pth"
    ckpt_path.write_bytes(b"dummy")  # allow SpyPredictor to see an existing path if needed

    monkeypatch.setattr(
        sys, "argv",
        ["prog",
         "--data-dir", str(tmp_path),   # irrelevant because we stub discover_pairs
         "--ckpt-path", str(ckpt_path),
         "--pred-dir", str(pred_dir),
         "--image-size", "8",
         "--device", "cpu"]
    )

    wire_pred_mocks.main()

    out = capsys.readouterr().out
    assert "[done] Wrote predictions to:" in out

    # The stub discover_pairs created 3 test files named test_0.png, test_1.png, test_2.png
    expect = [pred_dir / f"test_{i}.png" for i in range(3)]
    for p in expect:
        assert p.exists(), f"Missing predicted mask file: {p}"
        txt = p.read_text()
        assert "shape=" in txt  # written by fake_save_mask


def test_predict_raises_on_empty_test_set(monkeypatch, capsys, tmp_path):
    import dida_roofseg.predict as P

    # Stub discover_pairs to return 0 test images
    monkeypatch.setattr(
        "dida_roofseg.predict.discover_pairs",
        lambda data_dir: ([], {}, []),  # labeled, mask_map, test_images
        raising=True,
    )

    # Patch the rest minimally so import succeeds (won't be used because we error early)
    monkeypatch.setattr("dida_roofseg.predict.RoofDataset", DummyTestDataset, raising=True)
    dummy_model_mod = types.SimpleNamespace(
        EncoderWrapper=DummyEncoder,
        DecoderUNetSmall=DummyDecoder,
        SegmentationModel=DummySegmentationModel,
    )
    monkeypatch.setattr("dida_roofseg.predict.model_mod", dummy_model_mod, raising=True)
    monkeypatch.setattr("dida_roofseg.predict.Predictor", SpyPredictor, raising=True)
    monkeypatch.setattr("dida_roofseg.predict.save_mask", lambda mask, out: None, raising=True)

    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(RuntimeError, match="No test images found"):
        P.main()
