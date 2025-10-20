# src/dida_roofseg/viz.py

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


def save_masks(
    preds: Tensor,
    out_dir: str | Path,
    *,
    threshold: float | None = 0.5,
    prefix: str = "pred",
) -> list[Path]:
    """
    Save predicted masks as individual PNG files.

    Args:
        preds: (B,1,H,W) tensor. If float in [0,1], an optional threshold will binarize.
        out_dir: directory to write PNGs.
        threshold: if provided, binarize with (pred >= threshold).
        prefix: filename prefix.

    Returns:
        List of written file paths.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    preds = preds.detach().cpu()
    B = preds.size(0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for i in range(B):
        mask = preds[i, 0]
        if threshold is not None:
            mask = (mask >= float(threshold)).float()

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(mask.numpy(), interpolation="nearest")
        ax.axis("off")
        p = out_dir / f"{prefix}_{i:03d}.png"
        fig.savefig(p, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        written.append(p)

    return written


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

def main():
    # Build the same model shape used in training

    from dida_roofseg.dataset import RoofDataset
    from torch.utils.data import DataLoader
    from dida_roofseg.io import discover_pairs
    from dida_roofseg import model as model_mod
    from dida_roofseg.engine import Predictor

    labeled_images, mask_map, test_images = discover_pairs("data/raw")
    test_ds = RoofDataset(mode="val", image_paths=labeled_images, mask_dir_map=mask_map, image_size=512)
    #breakpoint()

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    encoder = model_mod.EncoderWrapper(name="resnet18", pretrained=False)
    decoder = model_mod.DecoderUNetSmall(encoder_channels=encoder.feature_channels)
    model = model_mod.SegmentationModel(encoder=encoder, decoder=decoder)

    predictor = Predictor(model=model, ckpt_path="models/checkpoints/best.pth")

    # create a batch of one image
    imgs,masks=next(iter(test_loader))
    preds=predictor.predict_batch(imgs)

    # create a batch of 5 images
    imgs_list=[]
    masks_list=[]
    for i,(imgs_batch,masks_batch) in enumerate(test_loader):
        imgs_list.append(imgs_batch)
        masks_list.append(masks_batch)
        if i>=4:
            break
    imgs=torch.cat(imgs_list,dim=0)
    masks=torch.cat(masks_list,dim=0)
    preds=predictor.predict_batch(imgs)

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
    
if __name__ == "__main__":
    main()

