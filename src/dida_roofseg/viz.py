# src/dida_roofseg/viz.py

"""
Utilities for visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor


def _to_hw3(
    x: Tensor,
    mean: Iterable[float] | None = None,
    std: Iterable[float] | None = None,
    clip: bool = True,
) -> np.ndarray:
    """
    Convert CHW or HW tensor to HxW or HxWx3 numpy image.
    Optionally de-normalizes with per-channel mean/std.
    """
    x = x.detach().cpu()
    if x.ndim == 3:
        # De-normalize if requested
        if mean is not None and std is not None and x.size(0) in (1, 3):
            m = torch.tensor(list(mean), dtype=x.dtype, device=x.device)
            s = torch.tensor(list(std), dtype=x.dtype, device=x.device)
            if x.size(0) == 1:  # grayscale case
                m = m[:1]
                s = s[:1]
            x = x * s.view(-1, 1, 1) + m.view(-1, 1, 1)

        if x.size(0) == 1:
            x = x[0]  # (H,W)
        elif x.size(0) >= 3:
            x = x[:3].permute(1, 2, 0)  # (H,W,3)
        else:
            # Fallback: replicate 1-channel to 3
            x = x[0].repeat(3, 1, 1).permute(1, 2, 0)
    elif x.ndim == 2:
        pass  # already HW
    else:
        raise ValueError(f"Expected tensor with 2 or 3 dims, got shape {tuple(x.shape)}")

    arr = x.numpy()
    if clip:
        arr = np.clip(arr, 0.0, 1.0)
    return arr


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with `pip install matplotlib`."
        ) from e


def plot_learning_curves(
    history: dict[str, list[float]],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """
    Plot learning curves from history dict.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    plt.plot(history["train_loss"], label="training loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.plot(history["val_iou"], label="validation IoU")
    plt.plot(history["val_dice"], label="validation dice")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(0.5, 0.5))
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_batch(
    imgs: Tensor,
    preds: Tensor,
    masks: Tensor | None = None,
    *,
    max_n: int = 5,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    overlay_alpha: float = 0.45,
    mean: Iterable[float] | None = None,
    std: Iterable[float] | None = None,
) -> None:
    """
    Plot a grid of up to `max_n` samples with columns:
    [Image | Prediction | (optional) Ground Truth].

    Args:
        imgs: (B,C,H,W) or (B,1,H,W) float tensor (assumed ~[0,1] or normalized).
        preds: (B,1,H,W) float/binary tensor in {0,1} or probabilities in [0,1].
        masks: optional (B,1,H,W) binary tensor in {0,1}.
        max_n: maximum rows to display.
        title: figure title.
        save_path: if provided, saves the PNG to this path.
        show: if True, calls plt.show().
        overlay_alpha: alpha for overlaying masks on top of the image.
        mean, std: optional per-channel de-normalization for imgs.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    imgs = imgs.detach().cpu()
    preds = preds.detach().cpu()
    masks = masks.detach().cpu() if masks is not None else None

    B = imgs.size(0)
    n = min(max_n, B)
    cols = 3 if masks is not None else 2

    fig, axes = plt.subplots(n, cols, figsize=(4 * cols, 4 * n))
    if n == 1 and cols == 1:
        axes = np.array([[axes]])
    elif n == 1:
        axes = np.array([axes])  # shape (1, cols)

    def _plot_row(i: int):
        # IMAGE
        ax = axes[i, 0] if cols > 1 else axes[i]
        ax.imshow(_to_hw3(imgs[i], mean=mean, std=std))
        ax.set_title("Image")
        ax.axis("off")

        # PREDICTION (overlay)
        ax = axes[i, 1] if cols > 1 else axes[i]
        ax.imshow(_to_hw3(imgs[i], mean=mean, std=std))
        # Ensure binary/float mask (H,W)
        pred_mask = preds[i, 0]
        # If probabilities, display as-is; overlay looks nicer with continuous alpha
        ax.imshow(pred_mask.numpy(), alpha=overlay_alpha)
        ax.set_title("Prediction")
        ax.axis("off")

        if cols == 3:
            ax = axes[i, 2]
            ax.imshow(_to_hw3(imgs[i], mean=mean, std=std))
            gt_mask = masks[i, 0]
            ax.imshow(gt_mask.numpy(), alpha=overlay_alpha)
            ax.set_title("Ground truth")
            ax.axis("off")

    for i in range(n):
        _plot_row(i)

    if title:
        fig.suptitle(title, y=0.995)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def dummy_plot():
    # Simple test
    import torch

    B, C, H, W = 2, 3, 128, 128
    imgs = torch.rand(B, C, H, W)
    preds = torch.rand(B, 1, H, W)
    masks = (torch.rand(B, 1, H, W) > 0.5).float()

    plot_batch(
        imgs,
        preds,
        masks,
        max_n=2,
        title="Test Plot",
        save_path="test_plot.png",
        show=False,
        overlay_alpha=0.5,
    )

    save_masks(preds, out_dir="test_masks", threshold=0.5, prefix="test_pred")
    
if __name__ == "__main__":
    import json
    # load the history.json file
    with open("models/checkpoints/history.json", "r") as f:
        history = json.load(f)

    # plot the learning curves
    plot_learning_curves(history, save_path="outputs/learning_curves.png", show=True)
    

