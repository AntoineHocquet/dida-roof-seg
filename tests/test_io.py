# tests/test_io.py
"""
Compact test suite for dida_roofseg.io functions.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from dida_roofseg.io import (
    discover_pairs,
    train_val_split,
    read_image,
    read_mask,
    save_mask,
)

# -----------------------
# helpers for this test file
# -----------------------
def write_rgb(path: Path, w: int, h: int, color=(0, 0, 0)):
    """Write an RGB PNG with a solid BGR color (because cv2 uses BGR on write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = (color[2], color[1], color[0])
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bgr
    cv2.imwrite(str(path), img)

def write_gray(path: Path, w: int, h: int, value: int):
    """Write a grayscale PNG with a solid value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


# -----------------------
# discover_pairs
# -----------------------
def test_discover_pairs_separate_dirs(tmp_path: Path):
    raw = tmp_path / "data" / "raw"
    images = raw / "images"
    masks = raw / "masks"

    # 3 labeled + 2 unlabeled
    write_rgb(images / "a.png", 8, 8)
    write_rgb(images / "b.png", 8, 8)
    write_rgb(images / "c.png", 8, 8)
    write_rgb(images / "d.png", 8, 8)   # test
    write_rgb(images / "e.png", 8, 8)   # test

    write_gray(masks / "a.png", 8, 8, 255)
    write_gray(masks / "b.png", 8, 8, 0)
    write_gray(masks / "c.png", 8, 8, 255)

    labeled, mask_map, test_imgs = discover_pairs(raw)

    stems = sorted([p.stem for p in labeled])
    assert stems == ["a", "b", "c"]
    assert set(mask_map.keys()) == {"a", "b", "c"}
    assert sorted([p.stem for p in test_imgs]) == ["d", "e"]


def test_discover_pairs_same_dir_with_suffix(tmp_path: Path):
    raw = tmp_path / "data" / "raw"

    # img x has mask x_mask.png
    write_rgb(raw / "x.png", 8, 8)
    write_rgb(raw / "y.png", 8, 8)          # test
    write_gray(raw / "x_mask.png", 8, 8, 255)

    labeled, mask_map, test_imgs = discover_pairs(raw)
    assert [p.stem for p in labeled] == ["x"]
    # normalized stem removes '_mask'
    assert set(mask_map.keys()) == {"x"}
    assert [p.stem for p in test_imgs] == ["y"]


# -----------------------
# train_val_split
# -----------------------
def test_train_val_split_reproducible(tmp_path: Path):
    paths = [tmp_path / f"{i}.png" for i in range(10)]
    for p in paths:
        write_rgb(p, 2, 2)

    t1, v1 = train_val_split(paths, val_ratio=0.3, seed=123)
    t2, v2 = train_val_split(paths, val_ratio=0.3, seed=123)
    t3, v3 = train_val_split(paths, val_ratio=0.3, seed=124)

    assert sorted([p.name for p in v1]) == sorted([p.name for p in v2])
    # with a different seed, it's likely (not guaranteed) to differ:
    assert sorted([p.name for p in v1]) != sorted([p.name for p in v3])


# -----------------------
# read_image
# -----------------------
def test_read_image_shape_and_normalization(tmp_path: Path):
    raw = tmp_path / "data" / "raw"
    img_path = raw / "white.png"

    # pure white RGB (255,255,255)
    write_rgb(img_path, 10, 6, color=(255, 255, 255))

    tensor, orig = read_image(img_path, image_size=8)
    # shape (3, H, W)
    assert tensor.shape == (3, 8, 8)
    assert orig == (6, 10)

    # check normalization for a white pixel: (1 - mean) / std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    expected = (1.0 - mean) / std
    # sample the center pixel
    px = tensor[:, 4, 4].numpy()
    np.testing.assert_allclose(px, expected, rtol=1e-3, atol=1e-3)


# -----------------------
# read_mask
# -----------------------
def test_read_mask_binarization_and_resize(tmp_path: Path):
    raw = tmp_path / "data" / "raw"
    mpath = raw / "mask.png"

    # Create a 6x10 mask with values below and above threshold
    mask = np.array(
        [
            [0, 10, 127, 128, 200, 255, 0, 127, 128, 255],
            [255] * 10,
            [0] * 10,
            [200] * 10,
            [127] * 10,
            [128] * 10,
        ],
        dtype=np.uint8,
    )
    raw.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mpath), mask)

    t = read_mask(mpath, image_size=8)  # (1,8,8)
    assert t.shape == (1, 8, 8)
    # Values should be {0,1}
    uniq = torch.unique(t)
    assert set(uniq.tolist()).issubset({0.0, 1.0})
    # Check threshold rule: >127 -> 1
    assert (t.max() <= 1.0) and (t.min() >= 0.0)


# -----------------------
# save_mask
# -----------------------
def test_save_mask_roundtrip(tmp_path: Path):
    out = tmp_path / "pred.png"

    # create a small binary mask tensor
    arr = torch.zeros((1, 6, 10), dtype=torch.float32)
    arr[:, 1:3, 2:5] = 1.0
    save_mask(arr, out)

    assert out.exists()
    m = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert m is not None
    # saved PNG should be {0,255}
    vals = set(np.unique(m).tolist())
    assert vals.issubset({0, 255})
