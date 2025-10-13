from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# -------------------------
# Losses and metrics
# -------------------------
def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * torch.sum(probs * targets) + eps
    den = torch.sum(probs) + torch.sum(targets) + eps
    dice = num / den
    return 1.0 - dice


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return bce_loss(logits, targets) + dice_loss(logits, targets)


@torch.no_grad()
def compute_iou(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    # preds/targets expected binary in {0,1}
    inter = torch.sum(preds * targets).item()
    union = torch.sum((preds + targets) >= 1).item()
    return float((inter + eps) / (union + eps))


@torch.no_grad()
def compute_dice(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    inter = torch.sum(preds * targets).item()
    s = torch.sum(preds).item() + torch.sum(targets).item()
    return float((2 * inter + eps) / (s + eps))


# -------------------------
# Trainer / Predictor
# -------------------------
@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 4
    lr_encoder: float = 1e-4
    lr_decoder: float = 1e-3
    weight_decay: float = 1e-4
    freeze_encoder_epochs: int = 3
    threshold: float = 0.5
    device: str = "cpu"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
        ckpt_dir: str | Path,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.ckpt_dir / "best.pth"
        self.best_iou = -np.inf

        # separate parameter groups (encoder vs. decoder) if exposed
        encoder_params = []
        decoder_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "encoder" in name:
                encoder_params.append(p)
            else:
                decoder_params.append(p)

        if len(encoder_params) == 0 or len(decoder_params) == 0:
            # fallback: single group
            params = [{"params": self.model.parameters(), "lr": self.cfg.lr_decoder, "weight_decay": self.cfg.weight_decay}]
        else:
            params = [
                {"params": encoder_params, "lr": self.cfg.lr_encoder, "weight_decay": self.cfg.weight_decay},
                {"params": decoder_params, "lr": self.cfg.lr_decoder, "weight_decay": self.cfg.weight_decay},
            ]

        self.opt = torch.optim.Adam(params)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.cfg.epochs)

    def _train_one_epoch(self, epoch: int, frozen: bool) -> float:
        self.model.train()
        total_loss = 0.0
        for imgs, masks in tqdm(self.train_loader, desc=f"Epoch {epoch+1} [{'FROZEN' if frozen else 'FT'}]"):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            self.opt.zero_grad(set_to_none=True)
            logits = self.model(imgs)
            loss = bce_dice_loss(logits, masks)
            loss.backward()
            self.opt.step()
            total_loss += loss.item() * imgs.size(0)
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        ious = []
        dices = []
        th = self.cfg.threshold
        for imgs, masks in self.val_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            logits = self.model(imgs)
            loss = bce_dice_loss(logits, masks)
            total_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= th).float()
            ious.append(compute_iou(preds, masks))
            dices.append(compute_dice(preds, masks))

        val_loss = total_loss / len(self.val_loader.dataset)
        iou = float(np.mean(ious)) if ious else 0.0
        dice = float(np.mean(dices)) if dices else 0.0
        return val_loss, iou, dice

    def fit(self) -> Path:
        # Try to freeze encoder parameters during warmup if encoder exposes .freeze()
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "freeze"):
            self.model.encoder.freeze()

        for epoch in range(self.cfg.epochs):
            frozen_phase = epoch < self.cfg.freeze_encoder_epochs
            if not frozen_phase and hasattr(self.model, "encoder") and hasattr(self.model.encoder, "unfreeze"):
                # ensure unfrozen after warmup
                self.model.encoder.unfreeze()

            train_loss = self._train_one_epoch(epoch, frozen=frozen_phase)
            val_loss, val_iou, val_dice = self._validate()
            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{self.cfg.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
                f"| val_iou={val_iou:.4f} | val_dice={val_dice:.4f}"
            )

            if val_iou > self.best_iou:
                self.best_iou = val_iou
                torch.save({"state_dict": self.model.state_dict()}, self.best_path)
                print(f"  ✔ Saved new best checkpoint: {self.best_path} (IoU={val_iou:.4f})")

        return self.best_path


class Predictor:
    def __init__(self, model: nn.Module, ckpt_path: str | Path, device: str = "cpu", threshold: float = 0.5):
        self.model = model
        self.device = torch.device(device)
        self.threshold = threshold
        self.model.to(self.device)
        self._load(ckpt_path)

    def _load(self, ckpt_path: str | Path) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, imgs: torch.Tensor) -> torch.Tensor:
        logits = self.model(imgs.to(self.device, non_blocking=True))
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).float()
        return preds
