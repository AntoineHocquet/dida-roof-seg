# tests/test_cli.py
"""
Merged tests for the unified CLI (dida_roofseg.cli) with subcommands:
- train
- predict

We monkeypatch dependencies so we can test flows without real data or training.
"""

import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from dida_roofseg.engine import TrainConfig
import types

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------- Dummies -------------------------------------

class DummyTrainDataset(Dataset):
    """Ignore real files; produce tiny tensors fast."""
    def __init__(self, mode, image_paths, mask_dir_map, image_size=None):
        self._n = len(image_paths)
        self.H = self.W = 8 if image_size is None else min(8, int(image_size))
    def __len__(self):
        return self._n
    def __getitem__(self, idx):
        x = torch.zeros(3, self.H, self.W, dtype=torch.float32)
        y = torch.zeros(1, self.H, self.W, dtype=torch.float32)
        return x, y

class DummyTestDataset(Dataset):
    """For predict: returns (image, meta) with filename in meta."""
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
        # matches what cli.build_model -> DecoderUNetSmall expects
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
        return torch.zeros(x.size(0), 1, x.size(2), x.size(3))

class SpyTrainer:
    """
    Captures cfg/loader/ckpt_dir; pretends to train and writes a fake ckpt.
    """
    last_init: dict[str, Any] | None = None
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: "TrainConfig",
        ckpt_dir: Path,
    ) -> None:
        type(self).last_init = {
            "model": model,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "cfg": cfg,
            "ckpt_dir": Path(ckpt_dir),
        }
        self.ckpt_dir = Path(ckpt_dir)
    def fit(self) -> str:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        out = self.ckpt_dir / "best.pth"
        out.write_bytes(b"dummy")
        return str(out)

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
        B, _, H, W = imgs.shape
        return torch.ones(B, 1, H, W, dtype=torch.float32)


# ----------------------------- Fixtures ------------------------------------

@pytest.fixture
def wire_cli_train_mocks(monkeypatch, tmp_path):
    """
    Monkeypatch dida_roofseg.cli for the `train` subcommand.
    """
    import dida_roofseg.cli as CLI

    # 1) Fake discover_pairs -> 25 labeled + 5 test files
    labeled = [tmp_path / f"img_{i}.png" for i in range(25)]
    test = [tmp_path / f"test_{i}.png" for i in range(5)]
    mask_map = {"whatever": tmp_path / "masks"}
    monkeypatch.setattr(
        "dida_roofseg.cli.discover_pairs",
        lambda data_dir: (labeled, mask_map, test),
        raising=True,
    )

    # 2) Use our dummy Dataset
    monkeypatch.setattr("dida_roofseg.cli.RoofDataset", DummyTrainDataset, raising=True)

    # 3) Swap out model module
    dummy_model_mod = types.SimpleNamespace(
        EncoderWrapper=DummyEncoder,
        DecoderUNetSmall=DummyDecoder,
        SegmentationModel=DummySegmentationModel,
    )
    monkeypatch.setattr("dida_roofseg.cli.model_mod", dummy_model_mod, raising=True)

    # 4) Swap out Trainer
    monkeypatch.setattr("dida_roofseg.cli.Trainer", SpyTrainer, raising=True)

    return CLI  # patched module under test


@pytest.fixture
def wire_cli_predict_mocks(monkeypatch, tmp_path):
    """
    Monkeypatch dida_roofseg.cli for the `predict` subcommand.
    """
    import dida_roofseg.cli as CLI

    # 1) Stub discover_pairs to provide only test images
    test_imgs = [tmp_path / f"test_{i}.png" for i in range(3)]
    labeled = []
    mask_map = {}
    monkeypatch.setattr(
        "dida_roofseg.cli.discover_pairs",
        lambda data_dir: (labeled, mask_map, test_imgs),
        raising=True,
    )

    # 2) Use tiny test dataset
    monkeypatch.setattr("dida_roofseg.cli.RoofDataset", DummyTestDataset, raising=True)

    # 3) Swap out models
    dummy_model_mod = types.SimpleNamespace(
        EncoderWrapper=DummyEncoder,
        DecoderUNetSmall=DummyDecoder,
        SegmentationModel=DummySegmentationModel,
    )
    monkeypatch.setattr("dida_roofseg.cli.model_mod", dummy_model_mod, raising=True)

    # 4) Swap out Predictor
    monkeypatch.setattr("dida_roofseg.cli.Predictor", SpyPredictor, raising=True)

    # 5) Replace save_mask to write a small file so we can assert on outputs
    def fake_save_mask(mask: torch.Tensor, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(f"shape={tuple(mask.shape)}")

    monkeypatch.setattr("dida_roofseg.cli.save_mask", fake_save_mask, raising=True)

    return CLI  # patched module under test


# ------------------------------- Tests -------------------------------------

def test_parse_defaults_train(monkeypatch):
    """
    Ensure default values for the 'train' subcommand.
    """
    from dida_roofseg.cli import build_parser
    monkeypatch.setattr(sys, "argv", ["prog", "train"])
    args = build_parser().parse_args()
    assert args.epochs == 20
    assert args.batch_size == 4
    assert args.lr_encoder == 1e-4
    assert args.lr_decoder == 1e-3
    assert args.encoder == "resnet18"
    assert args.device == "cpu"
    assert args.image_size == 512
    assert args.threshold == 0.5
    assert args.seed == 42

def test_parse_defaults_predict(monkeypatch):
    """
    Ensure default values for the 'predict' subcommand.
    """
    from dida_roofseg.cli import build_parser
    monkeypatch.setattr(sys, "argv", ["prog", "predict"])
    args = build_parser().parse_args()
    assert args.data_dir == "data/raw"
    assert args.ckpt_path == "models/checkpoints/best.pth"
    assert args.pred_dir == "outputs/predictions"
    assert args.image_size == 512
    assert args.encoder == "resnet18"
    assert args.threshold == 0.5
    assert args.seed == 42
    assert args.device == "cpu"

def test_train_main_runs_with_mocks(wire_cli_train_mocks, monkeypatch, capsys, tmp_path):
    """
    Run 'train' subcommand end-to-end with spies and verify side-effects.
    """
    ckpt_dir = tmp_path / "ckpts"
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "train",
         "--epochs", "1",
         "--batch-size", "1",
         "--image-size", "8",
         "--ckpt-dir", str(ckpt_dir),
         "--device", "cpu"]
    )
    wire_cli_train_mocks.main()
    out = capsys.readouterr().out
    assert "[done] Best checkpoint saved at:" in out
    assert (ckpt_dir / "best.pth").exists()

    # Sanity: loaders built over our stubbed 25 images split by default val_ratio=0.2
    init = SpyTrainer.last_init
    train_len = len(init["train_loader"].dataset)
    val_len = len(init["val_loader"].dataset)
    assert train_len + val_len == 25
    assert val_len == 5  # 25 * 0.2

def test_train_warns_when_test_count_not_five(monkeypatch, capsys, tmp_path):
    """
    Keep behavior: warn if the discovered test set size != 5.
    """
    import dida_roofseg.cli as CLI

    labeled = [tmp_path / f"img_{i}.png" for i in range(25)]
    test = [tmp_path / "oops.png"]  # only 1 test file
    mask_map = {"whatever": tmp_path / "masks"}
    monkeypatch.setattr("dida_roofseg.cli.discover_pairs",
                        lambda data_dir: (labeled, mask_map, test), raising=True)

    monkeypatch.setattr("dida_roofseg.cli.RoofDataset", DummyTrainDataset, raising=True)
    dummy_model_mod = types.SimpleNamespace(
        EncoderWrapper=DummyEncoder,
        DecoderUNetSmall=DummyDecoder,
        SegmentationModel=DummySegmentationModel,
    )
    monkeypatch.setattr("dida_roofseg.cli.model_mod", dummy_model_mod, raising=True)
    monkeypatch.setattr("dida_roofseg.cli.Trainer", SpyTrainer, raising=True)

    monkeypatch.setattr(sys, "argv", ["prog", "train", "--epochs", "1", "--batch-size", "1"])
    CLI.main()
    out = capsys.readouterr().out
    assert "[warn] Expected 5 test images without masks; found 1." in out

def test_train_cli_flags_flow_into_train_config(wire_cli_train_mocks, monkeypatch, tmp_path):
    """
    Ensure CLI flags propagate into TrainConfig in SpyTrainer.
    """
    ckpt_dir = tmp_path / "ck"
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "train",
         "--epochs", "3",
         "--batch-size", "7",
         "--lr-encoder", "0.0002",
         "--lr-decoder", "0.0015",
         "--weight-decay", "0.0003",
         "--freeze-epochs", "2",
         "--threshold", "0.42",
         "--val-ratio", "0.4",
         "--image-size", "16",
         "--encoder", "resnet34",
         "--device", "cpu",
         "--ckpt-dir", str(ckpt_dir)]
    )
    wire_cli_train_mocks.main()
    captured = SpyTrainer.last_init["cfg"]
    assert captured.epochs == 3
    assert captured.batch_size == 7
    assert captured.lr_encoder == pytest.approx(2e-4)
    assert captured.lr_decoder == pytest.approx(1.5e-3)
    assert captured.weight_decay == pytest.approx(3e-4)
    assert captured.freeze_encoder_epochs == 2
    assert captured.threshold == pytest.approx(0.42)
    assert captured.device == "cpu"

def test_predict_main_writes_files_and_prints(wire_cli_predict_mocks, monkeypatch, capsys, tmp_path):
    """
    Run 'predict' subcommand end-to-end with spies and verify outputs.
    """
    pred_dir = tmp_path / "preds"
    ckpt_path = tmp_path / "best.pth"
    ckpt_path.write_bytes(b"dummy")  # optional, SpyPredictor doesn't read, but realistic

    monkeypatch.setattr(
        sys, "argv",
        ["prog", "predict",
         "--data-dir", str(tmp_path),   # irrelevant due to stubbed discover_pairs
         "--ckpt-path", str(ckpt_path),
         "--pred-dir", str(pred_dir),
         "--image-size", "8",
         "--device", "cpu"]
    )

    wire_cli_predict_mocks.main()
    out = capsys.readouterr().out
    assert "[done] Wrote predictions to:" in out

    # The stub discover_pairs created 3 test files
    expect = [pred_dir / f"test_{i}.png" for i in range(3)]
    for p in expect:
        assert p.exists(), f"Missing predicted mask file: {p}"
        txt = p.read_text()
        assert "shape=" in txt  # written by fake_save_mask

def test_predict_raises_on_empty_test_set(monkeypatch, tmp_path):
    """
    Keep behavior: raise RuntimeError if no test images are discovered.
    """
    import dida_roofseg.cli as CLI

    # Stub discover_pairs to return zero test images
    monkeypatch.setattr(
        "dida_roofseg.cli.discover_pairs",
        lambda data_dir: ([], {}, []),
        raising=True,
    )

    # Patch the rest minimally so import/argparse succeed (won't be used due to early error)
    monkeypatch.setattr("dida_roofseg.cli.RoofDataset", DummyTestDataset, raising=True)
    dummy_model_mod = types.SimpleNamespace(
        EncoderWrapper=DummyEncoder,
        DecoderUNetSmall=DummyDecoder,
        SegmentationModel=DummySegmentationModel,
    )
    monkeypatch.setattr("dida_roofseg.cli.model_mod", dummy_model_mod, raising=True)
    monkeypatch.setattr("dida_roofseg.cli.Predictor", SpyPredictor, raising=True)
    monkeypatch.setattr("dida_roofseg.cli.save_mask", lambda mask, out: None, raising=True)

    monkeypatch.setattr(sys, "argv", ["prog", "predict"])
    with pytest.raises(RuntimeError, match="No test images found"):
        CLI.main()
