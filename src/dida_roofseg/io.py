# src/dida_roofseg/io.py
"""
I/O utilities for the DIDA roof segmentation project.

This module provides a **consistent, torch-first** pipeline for all in-memory
resizing operations while keeping OpenCV (`cv2`) for disk I/O (reading/writing).

What lives here:
1) **Discovery / listing**
   - list_files(...)           → recursively list files
   - discover_pairs(...)       → match images ↔ masks; return labeled + test images
   - train_val_split(...)      → reproducible filename-based split

2) **Tensor I/O utilities**
   - read_image(...)           → CHW float tensor normalized to ImageNet stats (+ original size)
   - read_mask(...)            → (1,H,W) float/binary tensor {0,1}
   - resize_tensor_image(...)  → torch-based bilinear resize for images
   - resize_tensor_mask(...)   → torch-based nearest-neighbor resize for masks
   - resize_mask_to(...)       → alias of resize_tensor_mask for convenience

3) **Saving / visualization**
   - save_mask(...)            → write predicted mask; optionally save a side-by-side
                                panel with (image | mask) or an overlay on the right.

Design choices:
- **OpenCV** is used only for disk I/O and color conversions when writing PNGs.
- **All in-memory resizes** (down or up) are performed with **torch.nn.functional.interpolate**.
  This ensures one code path (and enables easy GPU acceleration if tensors live on CUDA).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

# Supported image extensions
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Default normalization (ImageNet)
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def _is_image(p: Path) -> bool:
    """Return True if file extension looks like an image."""
    return p.suffix.lower() in IMG_EXTS


def _stem_without_mask_suffix(stem: str) -> str:
    """Normalize common naming by mapping 'xxx_mask' → 'xxx'."""
    return stem[:-5] if stem.endswith("_mask") else stem


# ---------------------------------------------------------------------------
# Discovery / listing
# ---------------------------------------------------------------------------

def list_files(root: str | Path) -> list[Path]:
    """
    Recursively list all files within a directory, sorted for determinism.
    """
    root = Path(root)
    return [p for p in sorted(root.rglob("*")) if p.is_file()]


def discover_pairs(raw_dir: str | Path) -> tuple[list[Path], dict[str, Path], list[Path]]:
    """
    Discover image/mask pairs and 'test' images (i.e., images without masks).

    Returns:
      labeled_images: list[Path]  — images that have a corresponding mask
      mask_map:       dict[str, Path]  — stem → mask path
      test_images:    list[Path]  — images without a mask

    Matching strategy:
      1) If 'raw_dir/images' and 'raw_dir/masks' exist, match by stem.
      2) Else, search the whole 'raw_dir':
         - heuristically consider files as masks if filename contains 'mask'
           or they are under a directory named 'mask|masks|labels'.
         - others are considered candidate images.
    """
    raw_dir = Path(raw_dir)
    images_dir = raw_dir / "images"
    masks_dir = raw_dir / "masks"

    if images_dir.exists() and masks_dir.exists():
        img_paths = [p for p in sorted(images_dir.iterdir()) if p.is_file() and _is_image(p)]
        mask_paths = [p for p in sorted(masks_dir.iterdir()) if p.is_file() and _is_image(p)]
    else:
        all_files = [p for p in list_files(raw_dir) if _is_image(p)]
        img_paths: list[Path] = []
        mask_paths: list[Path] = []
        for p in all_files:
            lower = p.name.lower()
            if "mask" in lower or p.parent.name.lower() in {"mask", "masks", "labels"}:
                mask_paths.append(p)
            else:
                img_paths.append(p)

    # Build mapping stem -> mask path (normalize '_mask' suffix)
    mask_map: dict[str, Path] = {}
    for m in mask_paths:
        stem = _stem_without_mask_suffix(m.stem)
        mask_map[stem] = m

    labeled_images: list[Path] = []
    test_images: list[Path] = []
    for im in img_paths:
        norm_stem = _stem_without_mask_suffix(im.stem)
        if norm_stem in mask_map:
            labeled_images.append(im)
        else:
            test_images.append(im)

    return labeled_images, mask_map, test_images


def train_val_split(paths: list[Path], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[Path], list[Path]]:
    """
    Reproducible train/val split based on filename order + RNG shuffle.

    - Sorts by filename first (to avoid hidden ordering biases across OS/filesystems).
    - Shuffles indices using a fixed seed.
    - Ensures at least one sample in val if possible.
    """
    paths = sorted(paths, key=lambda p: p.name)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_val = max(1, int(len(paths) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    train = [p for i, p in enumerate(paths) if i not in val_idx]
    val = [p for i, p in enumerate(paths) if i in val_idx]
    return train, val


# ---------------------------------------------------------------------------
# Torch-first resizing helpers
# ---------------------------------------------------------------------------

def resize_tensor_image(img_chw: Tensor, size_hw: tuple[int, int]) -> Tensor:
    """
    Resize a normalized image tensor **in-memory** using torch.
    - Input:  (C,H,W) with C ∈ {1,3}; assumed normalized already.
    - Output: (C,h,w) (same dtype/device as input)
    - Mode:   bilinear, align_corners=False (good default for natural images).
    """
    assert img_chw.dim() == 3 and img_chw.size(0) in (1, 3), "resize_tensor_image expects (C,H,W) with C in {1,3}"
    img_4d = img_chw.unsqueeze(0)  # (1,C,H,W)
    out = F.interpolate(img_4d, size=size_hw, mode="bilinear", align_corners=False)
    return out.squeeze(0)


def resize_tensor_mask(mask_chw: Tensor, size_hw: tuple[int, int]) -> Tensor:
    """
    Resize a binary/label mask tensor **in-memory** using torch.
    - Input:  (1,H,W)
    - Output: (1,h,w)
    - Mode:   nearest (preserve labels / avoid smoothing).
    """
    assert mask_chw.dim() == 3 and mask_chw.size(0) == 1, "resize_tensor_mask expects (1,H,W)"
    m_4d = mask_chw.unsqueeze(0)  # (1,1,H,W)
    out = F.interpolate(m_4d, size=size_hw, mode="nearest")
    return out.squeeze(0)


# Backward-compatible alias (kept for existing call sites)
def resize_mask_to(mask_tensor: Tensor, size_hw: tuple[int, int]) -> Tensor:
    """Alias for resize_tensor_mask(mask_tensor, size_hw)."""
    return resize_tensor_mask(mask_tensor, size_hw)


# ---------------------------------------------------------------------------
# Reading utilities (OpenCV for disk I/O, Torch for in-memory resize)
# ---------------------------------------------------------------------------

def read_image(path: str | Path, image_size: int | None = None) -> tuple[Tensor, tuple[int, int]]:
    """
    Read an RGB image from disk with OpenCV, convert to **normalized** torch tensor.

    Returns:
        img_chw:   (3,H,W) float32 torch tensor, normalized to ImageNet mean/std
        orig_size: (H0,W0) original size (before any resizing)

    Notes:
    - We keep **cv2** for disk I/O and BGR→RGB conversion (fast & robust).
    - If `image_size` is given, we perform the resize in **torch** with bilinear,
      preserving normalization and allowing GPU execution if tensor is moved later.
    """
    p = str(path)
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]

    # To tensor (C,H,W) in [0,1]
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # (3,H,W)
    tensor = torch.from_numpy(img)  # float32 CPU

    # Normalize (channel-wise)
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor - mean) / std

    # Optional resize in **torch**
    if image_size is not None:
        tensor = resize_tensor_image(tensor, (image_size, image_size))

    return tensor, (h0, w0)


def read_mask(path: str | Path, image_size: int | None = None) -> Tensor:
    """
    Read a mask from disk and return a **binary** torch tensor (1,H,W) with values in {0,1}.

    Steps:
    - Read with OpenCV as grayscale.
    - Binarize at 127 threshold.
    - Convert to torch tensor (1,H,W), dtype float32.
    - If `image_size` is provided, resize with **torch** using nearest-neighbor.
    """
    p = str(path)
    mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {p}")

    mask = (mask > 127).astype(np.float32)  # (H,W) in {0,1}
    mask = np.expand_dims(mask, axis=0)     # (1,H,W)
    tensor = torch.from_numpy(mask)         # float32 CPU

    if image_size is not None:
        tensor = resize_tensor_mask(tensor, (image_size, image_size))

    return tensor


# ---------------------------------------------------------------------------
# Saving / visualization
# ---------------------------------------------------------------------------

def _denorm_to_uint8(img_chw: Tensor, mean: tuple[float, float, float] = IMAGENET_MEAN,
                     std: tuple[float, float, float] = IMAGENET_STD) -> np.ndarray:
    """
    Convert a **normalized** torch tensor (C,H,W) to **uint8 RGB** numpy array (H,W,3).

    - If C=1, the channel is replicated to 3 for visualization.
    - Values are clamped to [0,1] before scaling to [0,255].
    """
    x = img_chw.detach().cpu().float()
    assert x.dim() == 3 and x.size(0) in (1, 3), "Expected (C,H,W) with C in {1,3}"
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    C, H, W = x.shape
    m = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    s = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    x = x * s + m                      # de-normalize
    x = torch.clamp(x, 0.0, 1.0)       # safety
    x = (x * 255.0).round().byte()     # [0,255]
    x = x.permute(1, 2, 0).numpy()     # (H,W,3) RGB uint8
    return x


def _mask_to_uint8(mask: Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    Convert a (H,W) or (1,H,W) mask tensor to uint8 {0,255} 2D numpy array (H,W).
    """
    m = mask
    if m.dim() == 3 and m.size(0) == 1:
        m = m[0]
    mask_np = (m.detach().cpu().float().numpy() >= threshold).astype(np.uint8) * 255
    return mask_np


def save_mask(
    mask_tensor: Tensor,
    out_path: str | Path,
    img_tensor: Tensor | None = None,
    *,
    overlay: bool = False,
    overlay_alpha: float = 0.45,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> None:
    """
    Save a predicted mask to PNG.

    Cases:
      - If `img_tensor` is **None**:
          Save the **binary mask** alone as grayscale PNG (values {0,255}).
      - If `img_tensor` is provided (normalized CHW):
          Save a side-by-side panel **(LEFT = RGB image | RIGHT = mask)**.
          If `overlay=True`, the RIGHT panel becomes a red overlay on top of the image.

    Notes:
      - This function does **not** resize tensors. Prepare correct sizes before calling.
      - Final write uses OpenCV. Any color panels are converted to BGR as cv2 expects.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mask_np = _mask_to_uint8(mask_tensor)  # (H,W), 0/255

    if img_tensor is None:
        # Save mask only (grayscale)
        cv2.imwrite(str(out_path), mask_np, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        return

    # Build side-by-side visualization
    img_rgb = _denorm_to_uint8(img_tensor, mean=mean, std=std)  # (H,W,3) RGB uint8

    if overlay:
        # Make a red layer from the mask and alpha-blend it over the image (on the RIGHT panel).
        red = np.zeros_like(img_rgb)
        red[..., 2] = mask_np  # put mask into red channel (RGB)
        # Convert to BGR for cv2 blending/writing
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        blended = cv2.addWeighted(img_bgr, 1.0, red, overlay_alpha, 0.0)
        right_bgr = blended
        left_bgr = img_bgr
    else:
        # Right panel as plain (3-channel) grayscale mask; left is the image itself.
        mask_bgr = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
        left_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        right_bgr = mask_bgr

    # Ensure same height before concatenation (defensive; normally shapes match)
    if left_bgr.shape[0] != right_bgr.shape[0]:
        h = min(left_bgr.shape[0], right_bgr.shape[0])
        # Keep aspect ratios; use INTER_AREA for image, INTER_NEAREST for masks/overlays
        left_bgr = cv2.resize(
            left_bgr, (int(left_bgr.shape[1] * h / left_bgr.shape[0]), h), interpolation=cv2.INTER_AREA
        )
        right_bgr = cv2.resize(
            right_bgr, (int(right_bgr.shape[1] * h / right_bgr.shape[0]), h), interpolation=cv2.INTER_NEAREST
        )

    panel = cv2.hconcat([left_bgr, right_bgr])
    cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_PNG_COMPRESSION, 9])
