# tests/test_train.py

"""
Unit tests for train.py
Note: We use pytest's monkeypatch to stub out dependencies of train.py
so we can test the CLI arg parsing and main() flow without
needing real data, real models, or real training.
"""

import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from dida_roofseg.engine import TrainConfig
import types
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytest


# --- Dummies ---------------------------------------------------------------

class DummyRoofDataset(Dataset):
    """Ignore files; produce tiny tensors fast."""
    def __init__(self, mode, image_paths, mask_dir_map, image_size=None):
        self._n = len(image_paths)
        self.H = self.W = 8 if image_size is None else min(8, int(image_size))
    def __len__(self):
        return self._n
    def __getitem__(self, idx):
        x = torch.zeros(3, self.H, self.W, dtype=torch.float32)
        y = torch.zeros(1, self.H, self.W, dtype=torch.float32)
        return x, y

class DummyEncoder(nn.Module):
    def __init__(self, name, pretrained): # in_channels
        super().__init__()
        self.d = nn.Parameter(torch.zeros(1))
        # mimic a realistic feature map channel list for the decoder
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
    def forward(self, x):  # never used because Trainer is mocked
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
            ckpt_dir: Path
        ) -> None:  # cfg, ckpt_dir):
        """
        Initialize the SpyTrainer.

        Args:
            model (nn.Module): The model to be trained..
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            cfg (TrainConfig): Training configuration parameters.
            ckpt_dir (Path): Directory to save checkpoints.
        """
        type(self).last_init = {
            "model": model,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "cfg": cfg, "ckpt_dir": Path(ckpt_dir),
        }
        self.ckpt_dir = Path(ckpt_dir)

    def fit(self) -> str:
        """
        Pretend to train the model and write a fake checkpoint.

        Returns:
            str: Path to the saved checkpoint.
        """
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        out = self.ckpt_dir / "best.pth"
        out.write_bytes(b"dummy")
        return str(out)

# --- Fixtures --------------------------------------------------------------

@pytest.fixture
def wire_mocks(monkeypatch, tmp_path):
    """Monkeypatch train.py dependencies with dummies and stub data discovery."""
    import dida_roofseg.train as T

    # 1) Fake discover_pairs -> 25 labeled + 5 test files
    labeled = [tmp_path / f"img_{i}.png" for i in range(25)]
    test = [tmp_path / f"test_{i}.png" for i in range(5)]
    mask_map = {"whatever": tmp_path / "masks"}

    monkeypatch.setattr(
        "dida_roofseg.train.discover_pairs",
        lambda data_dir: (labeled, mask_map, test),
        raising=True,
    )

    # 2) Keep deterministic split behavior but ensure it returns reproducible halves
    # (We can rely on real train_val_split, or stub)
    # Here we keep the real one.

    # 3) Swap out Dataset class
    monkeypatch.setattr("dida_roofseg.train.RoofDataset", DummyRoofDataset, raising=True)

    # 4) Swap out model module symbols
    dummy_model_mod = types.SimpleNamespace(
        EncoderWrapper=DummyEncoder,
        DecoderUNetSmall=DummyDecoder,
        SegmentationModel=DummySegmentationModel,
    )
    monkeypatch.setattr("dida_roofseg.train.model_mod", dummy_model_mod, raising=True)

    # 5) Swap out Trainer
    monkeypatch.setattr("dida_roofseg.train.Trainer", SpyTrainer, raising=True)

    return T  # the actual module under test

# --- Tests -----------------------------------------------------------------

def test_parse_args_defaults(monkeypatch):
    from dida_roofseg.train import parse_args
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = parse_args()
    assert args.epochs == 20
    assert args.batch_size == 4
    assert args.lr_encoder == 1e-4
    assert args.lr_decoder == 1e-3
    assert args.encoder == "resnet18"
    assert args.device == "cpu"

def test_main_runs_with_mocks(wire_mocks, monkeypatch, capsys, tmp_path):
    # Provide CLI with minimal runtime and tiny images
    ckpt_dir = tmp_path / "ckpts"
    monkeypatch.setattr(
        sys, "argv",
        ["prog",
         "--epochs", "1",
         "--batch-size", "1",
         "--image-size", "8",
         "--ckpt-dir", str(ckpt_dir),
         "--device", "cpu"]
    )
    wire_mocks.main()
    out = capsys.readouterr().out
    assert "[done] Best checkpoint saved at:" in out
    # Ensure SpyTrainer wrote the checkpoint
    assert (ckpt_dir / "best.pth").exists()
    # Sanity: loaders built over our stubbed 25 images split by default val_ratio=0.2
    init = SpyTrainer.last_init
    train_len = len(init["train_loader"].dataset)
    val_len = len(init["val_loader"].dataset)
    # 25 with val_ratio=0.2 -> 20 train, 5 val (your real splitter should match this)
    assert train_len + val_len == 25
    assert val_len in {5, 5}  # explicit on purpose

def test_warns_when_test_count_not_five(monkeypatch, tmp_path, capsys):
    import dida_roofseg.train as T

    # Stub discover_pairs to return wrong test size
    labeled = [tmp_path / f"img_{i}.png" for i in range(25)]
    test = [tmp_path / "oops.png"]  # only 1 test file
    mask_map = {"whatever": tmp_path / "masks"}
    monkeypatch.setattr("dida_roofseg.train.discover_pairs",
                        lambda data_dir: (labeled, mask_map, test), raising=True)

    # Stub everything else minimally
    monkeypatch.setattr("dida_roofseg.train.RoofDataset", DummyRoofDataset, raising=True)
    dummy_model_mod = types.SimpleNamespace(
        EncoderWrapper=DummyEncoder,
        DecoderUNetSmall=DummyDecoder,
        SegmentationModel=DummySegmentationModel,
    )
    monkeypatch.setattr("dida_roofseg.train.model_mod", dummy_model_mod, raising=True)
    monkeypatch.setattr("dida_roofseg.train.Trainer", SpyTrainer, raising=True)

    monkeypatch.setattr(sys, "argv", ["prog", "--epochs", "1", "--batch-size", "1"])
    T.main()
    out = capsys.readouterr().out
    assert "[warn] Expected 5 test images without masks; found 1." in out

def test_cli_flags_flow_into_train_config(wire_mocks, monkeypatch, tmp_path):
    ckpt_dir = tmp_path / "ck"
    monkeypatch.setattr(
        sys, "argv",
        ["prog",
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
    wire_mocks.main()
    captured = SpyTrainer.last_init["cfg"]
    # Assert the CLI args made it into TrainConfig
    assert captured.epochs == 3
    assert captured.batch_size == 7
    assert captured.lr_encoder == pytest.approx(2e-4)
    assert captured.lr_decoder == pytest.approx(1.5e-3)
    assert captured.weight_decay == pytest.approx(3e-4)
    assert captured.freeze_encoder_epochs == 2
    assert captured.threshold == pytest.approx(0.42)
    assert captured.device == "cpu"
