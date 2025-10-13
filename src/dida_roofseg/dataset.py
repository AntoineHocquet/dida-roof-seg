from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .io import read_image, read_mask


class RoofDataset(Dataset):
    """
    OOP Dataset for train/val/test.

    train/val:
      - inputs: image_paths, mask_paths (aligned via stem)
      - returns: (image_tensor: (3,H,W), mask_tensor: (1,H,W))

    test:
      - inputs: image_paths only
      - returns: (image_tensor: (3,H,W), meta: dict with 'filename','orig_size')
    """

    def __init__(
        self,
        mode: str,
        image_paths: List[Path],
        mask_dir_map: Optional[dict] = None,
        image_size: Optional[int] = None,
    ):
        assert mode in {"train", "val", "test"}
        self.mode = mode
        self.image_paths = image_paths
        self.mask_map = mask_dir_map or {}
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def _find_mask_path(self, image_path: Path) -> Optional[Path]:
        stem = image_path.stem
        # try exact stem
        if stem in self.mask_map:
            return self.mask_map[stem]
        # try removing "_mask"
        norm = stem[:-5] if stem.endswith("_mask") else stem
        return self.mask_map.get(norm, None)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img_tensor, orig_size = read_image(img_path, image_size=self.image_size)

        if self.mode in {"train", "val"}:
            mask_path = self._find_mask_path(img_path)
            if mask_path is None:
                raise RuntimeError(f"No mask found for {img_path.name}")
            mask_tensor = read_mask(mask_path, image_size=self.image_size)
            return img_tensor.float(), mask_tensor.float()
        else:
            meta = {
                "filename": img_path.name,
                "orig_size": orig_size,  # (H,W)
                "path": str(img_path),
            }
            return img_tensor.float(), meta
