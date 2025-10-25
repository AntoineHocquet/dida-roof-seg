# src/dida_roofseg/engine.py

"""
Engine for training and inference.
Description:
 This module provides Trainer and Predictor classes to handle model training and inference.
 It includes implementations of common losses and metrics for binary segmentation tasks.
 The 
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# -------------------------
# Losses and metrics
# logits: (B,1,H,W), preds/targets: expected (B,1,H,W) in {0,1}
# 0 = perfect, 1 = worst
# P=preds, T=targets and eps = small constant to avoid division by zero.
# -------------------------
def bce_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Binary Cross-Entropy with Logits loss.
    Formula: -sum(T*log(P) + (1-T)*log(1-P))
    """
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def dice_loss(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Dice loss (1 - Dice coefficient) for binary masks 
    Formula: 1 - 2*|P∩T|/(|P|+|T|) = 1 - (2*sum(P*T)+eps)/(sum(P)+sum(T)+eps)
    (this is a loss, so lower is better).
    """
    probs = torch.sigmoid(logits)
    num = 2.0 * torch.sum(probs * targets) + eps
    den = torch.sum(probs) + torch.sum(targets) + eps
    dice = num / den
    return 1.0 - dice


def bce_dice_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Combined BCE + Dice loss.
    """
    return bce_loss(logits, targets) + dice_loss(logits, targets)


@torch.no_grad() # disable grad for metrics
def compute_iou(preds: Tensor, targets: Tensor, eps: float = 1e-6) -> float:
    """
    Compute Intersection over Union (IoU) for binary masks.
    Formula: |P∩T|/|P∪T| = (sum(P*T)+eps)/(sum((P+T)>=1)+eps)
    (This is a performance indicator, so higher is better.)
    """
    inter = torch.sum(preds * targets).item()
    union = torch.sum((preds + targets) >= 1).item()
    return float((inter + eps) / (union + eps))


@torch.no_grad() # disable grad for metrics
def compute_dice(preds: Tensor, targets: Tensor, eps: float = 1e-6) -> float:
    """
    Compute Dice coefficient for binary masks.
    Formula: 2*|P∩T|/(|P|+|T|) = (2*sum(P*T)+eps)/(sum(P)+sum(T)+eps)
    (Performance indicator, higher is better.)
    """
    inter = torch.sum(preds * targets).item()
    s = torch.sum(preds).item() + torch.sum(targets).item()
    return float((2 * inter + eps) / (s + eps))


# -------------------------
# Trainer / Predictor
# -------------------------
@dataclass
class TrainConfig:
    """
    Configuration for training (hyperparameters).
    """
    epochs: int = 20
    batch_size: int = 4
    lr_encoder: float = 1e-4
    lr_decoder: float = 1e-3
    weight_decay: float = 1e-4
    freeze_encoder_epochs: int = 3
    threshold: float = 0.5
    device: str = "cpu"


class Trainer:
    """
    Trainer for the model (OOP- style).
    """
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
        self.best_path_iou = self.ckpt_dir / "best_iou.pth"
        self.best_path_dice = self.ckpt_dir / "best_dice.pth"
        self.last_path = self.ckpt_dir / "last.pth"
        self.history_path = self.ckpt_dir / "history.json"
        self.best_iou = -np.inf
        self.best_dice = -np.inf

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
        """
        Helper to train for one epoch.
        Setting `frozen` to True will freeze the encoder parameters (i.e. training will affect only the decoder).
        """
        self.model.train()
        total_loss = 0.0
        num_examples = 0

        for imgs, masks in tqdm(self.train_loader, desc=f"Epoch {epoch+1} [{'FROZEN' if frozen else 'FT'}]"):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            self.opt.zero_grad(set_to_none=True)
            logits = self.model(imgs)
            loss = bce_dice_loss(logits, masks)
            loss.backward()
            self.opt.step()

            bsz = imgs.size(0) # batch size (may be smaller on last batch)
            total_loss += loss.item() * bsz # sum loss over batch
            num_examples += bsz # count examples

        return total_loss / max(1, num_examples) # avoid div by zero

    @torch.no_grad()
    def _validate(self) -> tuple[float, float, float]:
        """
        Helper to validate on the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        num_examples = 0
        ious = []
        dices = []
        th = self.cfg.threshold

        for imgs, masks in self.val_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            logits = self.model(imgs)
            loss = bce_dice_loss(logits, masks)

            bsz = imgs.size(0)
            total_loss += loss.item() * bsz
            num_examples += bsz

            probs = torch.sigmoid(logits)
            preds = (probs >= th).float()
            ious.append(compute_iou(preds, masks))
            dices.append(compute_dice(preds, masks))

        val_loss = total_loss / max(1, num_examples)
        iou = float(np.mean(ious)) if ious else 0.0
        dice = float(np.mean(dices)) if dices else 0.0
        return val_loss, iou, dice

    def fit(self) -> tuple[Path, Path, Path, Path]:
        """
        Main training loop.
        Returns the paths to the best IoU and Dice checkpoints, as well as the last checkpoint and the training history.
        """
        # Try to freeze encoder parameters during warmup if encoder exposes .freeze()
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "freeze"):
            self.model.encoder.freeze()
        
        # initialize learning curves
        train_loss_curve = []
        val_loss_curve = []
        val_iou_curve = []
        val_dice_curve = []

        # Train (start loop)
        for epoch in range(self.cfg.epochs):
            frozen_phase = epoch < self.cfg.freeze_encoder_epochs
            if not frozen_phase and hasattr(self.model, "encoder") and hasattr(self.model.encoder, "unfreeze"):
                # ensure unfrozen after warmup
                self.model.encoder.unfreeze()

            # Compute losses
            train_loss = self._train_one_epoch(epoch, frozen=frozen_phase)
            val_loss, val_iou, val_dice = self._validate()

            # Update learning curves
            train_loss_curve.append(train_loss)
            val_loss_curve.append(val_loss)
            val_iou_curve.append(val_iou)
            val_dice_curve.append(val_dice)

            # Update learning rate
            self.scheduler.step()

            # Print progress
            print(
                f"Epoch {epoch+1}/{self.cfg.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
                f"| val_iou={val_iou:.4f} | val_dice={val_dice:.4f}"
            )

            # Save checkpoint if IoU has improved
            if val_iou > self.best_iou:
                self.best_iou = val_iou
                torch.save(self.model.state_dict(), self.best_path_iou)
                print(f"  ✔ Saved new best checkpoint: {self.best_path_iou} (IoU={val_iou:.4f})")

            # Save checkpoint if Dice has improved
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                torch.save(self.model.state_dict(), self.best_path_dice)
                print(f"  ✔ Saved new best checkpoint: {self.best_path_dice} (Dice={val_dice:.4f})")

        # Always save last checkpoint
        torch.save(self.model.state_dict(), self.last_path)
        print(f"  Saved last checkpoint: {self.last_path} (Epochs={self.cfg.epochs})")

        # Save learning curves as JSON
        history = {
            "train_loss": train_loss_curve,
            "val_loss": val_loss_curve,
            "val_iou": val_iou_curve,
            "val_dice": val_dice_curve,
        }
        with open(self.ckpt_dir / "history.json", "w") as f:
            json.dump(history, f)
            print(f"  Saved learning curves: {self.history_path}")


        return self.best_path_iou, self.best_path_dice, self.last_path, self.history_path


class Predictor:
    """
    Predictor for inference using a trained model.
    """
    def __init__(self, model: nn.Module, ckpt_path: str | Path, device: str = "cpu", threshold: float = 0.5):
        self.model = model
        self.device = torch.device(device)
        self.threshold = threshold
        self.model.to(self.device)
        self._load(ckpt_path)

    def _load(self, ckpt_path: str | Path) -> None:
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
    
    @torch.no_grad()
    def predict_batch(self, imgs: Tensor) -> Tensor:
        """
        Single convenience method to predict a batch of images.
        Args:
            imgs: (B,C,H,W) tensor of input images.
        Returns:
            (B,1,H,W) tensor of predicted binary masks.
        """
        logits = self.model(imgs.to(self.device, non_blocking=True))
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).float()
        return preds
