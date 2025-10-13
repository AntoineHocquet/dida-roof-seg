import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def _stem_without_mask_suffix(stem: str) -> str:
    # normalize "xxx_mask" -> "xxx" (common dataset naming)
    return stem[:-5] if stem.endswith("_mask") else stem


def list_files(root: str | Path) -> List[Path]:
    root = Path(root)
    return [p for p in sorted(root.rglob("*")) if p.is_file()]


def discover_pairs(raw_dir: str | Path) -> Tuple[List[Path], Dict[str, Path], List[Path]]:
    """
    Returns:
      - all image paths that look like inputs,
      - dict: stem -> mask path for labeled samples,
      - test image paths (no mask found)
    Matching rules:
      1) If data/raw/images and data/raw/masks exist: match by stem.
      2) Else: search in data/raw/ for images, and masks are either same-name
         or with '_mask' suffix.
    """
    raw_dir = Path(raw_dir)
    images_dir = raw_dir / "images"
    masks_dir = raw_dir / "masks"

    if images_dir.exists() and masks_dir.exists():
        img_paths = [p for p in sorted(images_dir.iterdir()) if p.is_file() and _is_image(p)]
        mask_paths = [p for p in sorted(masks_dir.iterdir()) if p.is_file() and _is_image(p)]
    else:
        all_files = [p for p in list_files(raw_dir) if _is_image(p)]
        # Heuristic: masks often include 'mask' in filename or reside in a 'mask' folder
        img_paths = []
        mask_paths = []
        for p in all_files:
            lower = p.name.lower()
            if "mask" in lower or p.parent.name.lower() in {"mask", "masks", "labels"}:
                mask_paths.append(p)
            else:
                img_paths.append(p)

    # build mapping stem->mask
    mask_map: Dict[str, Path] = {}
    for m in mask_paths:
        stem = _stem_without_mask_suffix(m.stem)
        mask_map[stem] = m

    labeled_images: List[Path] = []
    test_images: List[Path] = []
    for im in img_paths:
        stem = im.stem
        norm_stem = _stem_without_mask_suffix(stem)
        if norm_stem in mask_map:
            labeled_images.append(im)
        else:
            test_images.append(im)

    return labeled_images, mask_map, test_images


def train_val_split(paths: List[Path], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_val = max(1, int(len(paths) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    train = [p for i, p in enumerate(paths) if i not in val_idx]
    val = [p for i, p in enumerate(paths) if i in val_idx]
    return train, val


def read_image(path: str | Path, image_size: Optional[int] = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Returns a tensor (3,H,W) normalized to ImageNet mean/std, and the original (H,W).
    """
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]

    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # (3,H,W)
    tensor = torch.from_numpy(img)
    return tensor, (h0, w0)


def read_mask(path: str | Path, image_size: Optional[int] = None) -> torch.Tensor:
    """
    Returns a binary tensor (1,H,W) in {0,1}.
    """
    path = str(path)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")

    if image_size is not None:
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    mask = (mask > 127).astype(np.float32)  # binarize
    mask = np.expand_dims(mask, axis=0)  # (1,H,W)
    tensor = torch.from_numpy(mask)
    return tensor


def save_mask(mask_tensor: torch.Tensor, out_path: str | Path) -> None:
    """
    Save a binary mask tensor (H,W) or (1,H,W) to PNG with values {0,255}.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_tensor.dim() == 3 and mask_tensor.size(0) == 1:
        mask_np = mask_tensor.squeeze(0).detach().cpu().numpy()
    else:
        mask_np = mask_tensor.detach().cpu().numpy()
    mask_np = (mask_np > 0.5).astype(np.uint8) * 255
    cv2.imwrite(str(out_path), mask_np)
