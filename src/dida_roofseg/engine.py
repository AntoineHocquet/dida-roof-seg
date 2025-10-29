# src/dida_roofseg/engine.py

"""
Engine for training and inference.
Description:
 This module provides Trainer and Predictor classes to handle model training and inference.
 It includes implementations of common losses and metrics for binary segmentation tasks.
 The 
"""

from __future__ import annotations
from typing import Generator, Any, Iterator
from os import PathLike

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import RoofDataset


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
    Stateless *inference* logic is centralized into a single `_predict_batch()` method.
    The class provides three ergonomic front-ends:

      1) `predict_next_batch()`  — stateful stepping through the dataset (nice for demos/plots).
      2) iteration protocol      — `for preds, metas_or_masks in predictor: ...` (one full pass).
      3) `predict_with_inputs()` — yields `(imgs, preds, targets_or_metas)` (perfect for plotting).

    It also guarantees that `Predictor.mode` always matches `dataset.mode`, via `set_mode()`.
    No file I/O is performed here — keep saving/resizing in CLI or io.py utilities.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        ckpt_path: str | PathLike[str],
        dataset: RoofDataset,
        *,
        mode: str = "test",          # "train" | "val" | "test"
        device: str = "cpu",
        threshold: float = 0.5,
        batch_size: int = 1,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> None:
        # ---- core config ----
        assert mode in {"train", "val", "test"}, f"Unsupported mode: {mode}"
        self.model = model
        self.device = torch.device(device)
        self.threshold = float(threshold)

        # Move model once; load checkpoint; eval mode for inference
        self.model.to(self.device)
        self._load(ckpt_path)
        self.model.eval()

        # ---- dataset & loader ----
        self.dataset: RoofDataset = dataset
        self.mode: str = mode
        self.dataset.mode = mode  # keep in sync

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

        self.dataloader: DataLoader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        # Internal iterator for step-wise prediction
        self._iter: Iterator | None = None

    # -------------------------
    # Lifecycle / configuration
    # -------------------------

    def _load(self, ckpt_path: str | PathLike[str]) -> None:
        """
        Load a state_dict checkpoint strictly and put model in eval mode.
        """
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def set_mode(
        self,
        mode: str,
        dataset: RoofDataset | None = None,
        *,
        batch_size: int | None = None,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
    ) -> None:
        """
        Unified entry-point to switch between train/val/test splits AND (optionally) swap datasets.
        This guarantees Predictor.mode == dataset.mode and rebuilds the DataLoader accordingly.
        """
        assert mode in {"train", "val", "test"}, f"Unsupported mode: {mode}"

        if dataset is not None:
            self.dataset = dataset
        self.mode = mode
        self.dataset.mode = mode

        if batch_size is not None:
            self.batch_size = int(batch_size)
        if num_workers is not None:
            self.num_workers = int(num_workers)
        if pin_memory is not None:
            self.pin_memory = bool(pin_memory)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self._iter = None  # reset internal iterator after any reconfiguration

    def _assert_mode_sync(self) -> None:
        """
        Defensive check to catch accidental drift between predictor.mode and dataset.mode.
        """
        ds_mode = getattr(self.dataset, "mode", None)
        if ds_mode != self.mode:
            raise RuntimeError(
                f"Predictor.mode ('{self.mode}') != dataset.mode ('{ds_mode}'). "
                "Use predictor.set_mode(...) to keep them synchronized."
            )

    def reset(self) -> None:
        """
        Reset the internal iterator for step-wise prediction (predict_next_batch).
        """
        self._iter = iter(self.dataloader)

    # -------------------------
    # Core inference primitive
    # -------------------------

    @torch.no_grad()
    def _predict_batch(self, imgs: Tensor, threshold: float | None = None) -> Tensor:
        """
        Core inference: forward → sigmoid → threshold.
        Returns a binary tensor (B,1,H,W) on CPU.

        If you need probabilities for any post-processing, fork here to return `probs` as well.
        """
        logits = self.model(imgs.to(self.device, non_blocking=True))
        probs = torch.sigmoid(logits)
        th = self.threshold if threshold is None else float(threshold)
        preds = (probs >= th).float()
        return preds.cpu()

    # -------------------------
    # Stepping API (nice for demos)
    # -------------------------

    @torch.no_grad()
    def predict_next_batch(
        self, threshold: float | None = None
    ) -> tuple[Tensor, list[dict[str, Any]] | None]:
        """
        Step once through the dataset and return predictions for the *next* batch.
        Does NOT auto-reset on exhaustion: subsequent call will raise StopIteration.
        Call `reset()` before the first call (or let this method create one lazily).

        Returns:
          - preds: (B,1,H,W) binary tensor (CPU).
          - metas_or_none:
              * test mode → list[dict] with keys like 'filename', 'orig_size', 'path'.
              * train/val → None (targets are masks, not meta dicts).
        """
        self._assert_mode_sync()
        if self._iter is None:
            self.reset()

        batch = next(self._iter)  # may raise StopIteration
        if self.mode in {"train", "val"}:
            imgs, _masks = batch
            metas = None
        else:
            imgs, metas = batch  # metas: list[dict] per item

        preds = self._predict_batch(imgs, threshold=threshold)
        return preds, metas

    # -------------------------
    # Iteration protocol (one full pass)
    # -------------------------

    def __iter__(self) -> Generator[tuple[Tensor, list[dict[str, Any]] | None], None, None]:
        """
        Iterate exactly once over the current dataloader, yielding per-batch:
            (preds, metas_or_none)
        where `metas_or_none` is:
            - None for train/val (because the second element of the batch is a mask tensor),
            - list[dict] for test (filename, orig_size, ...).
        """
        self._assert_mode_sync()
        for batch in self.dataloader:
            if self.mode in {"train", "val"}:
                imgs, _masks = batch
                metas = None
            else:
                imgs, metas = batch
            preds = self._predict_batch(imgs, threshold=None)
            yield preds, metas

    @torch.no_grad()
    def predict_all(self, threshold: float | None = None) -> list[tuple[Tensor, list[dict[str, Any]] | None]]:
        """
        Convenience method that COLLECTS all `(preds, metas_or_none)` batches into a list.
        Uses the iterator above to avoid any code duplication.
        """
        out: list[tuple[Tensor, list[dict[str, Any]] | None]] = []
        for preds, metas in self:
            # If you want a per-call threshold override, adapt _predict_batch to return probs and re-threshold here.
            out.append((preds, metas))
        return out

    # -------------------------
    # Plotting-friendly generator
    # -------------------------

    @torch.no_grad()
    def predict_with_inputs(
        self, threshold: float | None = None
    ) -> Generator[tuple[Tensor, Tensor, Tensor | list[dict[str, Any]]], None, None]:
        """
        Yield triples tailored for visualization:

            TRAIN/VAL: (imgs, preds, masks)
            TEST:      (imgs, preds, metas)

        All tensors are on CPU. Use this with your `viz.plot_batch(...)`.

        Example:
            predictor.set_mode("train", dataset=train_ds, batch_size=4)
            for imgs, preds, masks in predictor.predict_with_inputs():
                plot_batch(imgs, preds, masks, ...)
                break
        """
        self._assert_mode_sync()
        for batch in self.dataloader:
            if self.mode in {"train", "val"}:
                imgs, masks = batch                      # masks: (B,1,H,W)
                preds = self._predict_batch(imgs, threshold=threshold)  # (B,1,H,W)
                yield imgs.cpu(), preds, masks.cpu()
            else:
                imgs, metas = batch                      # metas: list[dict]
                preds = self._predict_batch(imgs, threshold=threshold)
                yield imgs.cpu(), preds, metas

    # -------------------------
    # Training metrics (optional evaluation utility)
    # -------------------------

    @torch.no_grad()
    def yield_training_scores(
        self,
        threshold: float | None = None,
        )-> dict[str, float]:
        """
        Compute training-set scores (loss, IoU, Dice) for the current model.

        This function reuses the same dataset and DataLoader but enforces
        self.mode = "train". It iterates once through the loader and returns
        averaged metrics.

        Returns:
            dict with keys: {"train_loss", "train_iou", "train_dice"}
        """
        from .engine import bce_dice_loss, compute_iou, compute_dice # to avoid circular import
        
        self._assert_mode_sync()
        if self.mode != "train":
            raise RuntimeError(
                "yield_training_scores() should be called only in 'train' mode. "
                "Use predictor.set_mode('train', dataset=train_ds) first."
            )

        scores: dict[str, float] = {}
        total_loss = 0.0
        num_examples = 0
        ious: list[float] = []
        dices: list[float] = []
        th = self.threshold if threshold is None else float(threshold)

        for imgs, masks in self.dataloader:
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

        scores["train_loss"] = total_loss / max(1, num_examples)
        scores["train_iou"] = float(torch.tensor(ious).mean().item() if ious else 0.0)
        scores["train_dice"] = float(torch.tensor(dices).mean().item() if dices else 0.0)

        return scores