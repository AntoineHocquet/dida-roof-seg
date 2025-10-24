# src/dida_roofseg/dataset.py
"""
Dataset class for roof segmentation.
Description: 
 This module defines a PyTorch Dataset class to handle training, validation,
   and test datasets for roof segmentation tasks. It uses the utilities from io.py.
 The class supports loading images and corresponding masks, with optional resizing for model input.
"""

from pathlib import Path # for file paths and type hints
from torch import Tensor # only for type hints
from torch.utils.data import Dataset

from .io import read_image, read_mask

class RoofDataset(Dataset[tuple[Tensor, Tensor]]): 
    # generic base class (torch>=2.0): above type hint declares that each __getitem__returns a pair of tensors
    """
    OOP Dataset for train/val/test.
    It has 3 modes: train, val, test, working as follows:
    - train/val:
      - inputs: image_paths, mask_paths (aligned via stem)
      - returns: (image_tensor: (3,H,W), mask_tensor: (1,H,W))
    - test:
      - inputs: image_paths only
      - returns: (image_tensor: (3,H,W), meta: dict with 'filename','orig_size')
    """

    def __init__(
        self,
        mode: str,
        image_paths: list[Path],
        mask_dir_map: dict[str, Path] | None = None,
        image_size: int | None = None,
    ) -> None:
        """
        Args:
          - mode: "train", "val", or "test".
          - image_paths: list of image file paths
          - mask_dir_map: dict mapping image stem to mask path (required for train/val)
          - image_size: if given, resize images/masks to (image_size, image_size)
        """
        assert mode in {"train", "val", "test"}
        self.mode: str = mode
        self.image_paths: list[Path] = image_paths
        self.mask_map: dict[str, Path] = mask_dir_map or {}
        self.image_size: int | None = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def _find_mask_path(self, image_path: Path) -> Path | None:
        """Return the matching mask path for a given image path, or None if not found."""
        stem = image_path.stem
        # try exact stem
        if stem in self.mask_map:
            return self.mask_map[stem]
        # try removing "_mask"
        norm = stem[:-5] if stem.endswith("_mask") else stem
        return self.mask_map.get(norm, None)

    def __getitem__(
            self,
            idx: int
        ) -> tuple[Tensor, Tensor] | tuple[Tensor, dict[str, str | tuple[int, int]]]:
        """
        Dunder method to get a sample by index.
        For train/val, returns (image_tensor, mask_tensor).
        For test, returns (image_tensor, meta_dict).
        If the mask is not found for train/val, raises RuntimeError.
        """
        img_path = self.image_paths[idx]
        img_tensor, orig_size = read_image(img_path, image_size=self.image_size)

        if self.mode in {"train", "val"}:
            mask_path = self._find_mask_path(img_path)
            if mask_path is None:
                raise RuntimeError(f"No mask found for {img_path.name}")
            mask_tensor = read_mask(mask_path, image_size=self.image_size)
            return img_tensor.float(), mask_tensor.float()
        else:
            meta: dict[str, str | tuple[int, int]] = {
                "filename": img_path.name,
                "orig_size": orig_size,  # (H,W)
                "path": str(img_path),
            }
            return img_tensor.float(), meta
