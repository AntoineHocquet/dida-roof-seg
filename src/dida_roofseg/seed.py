# src/dida_roofseg/seed.py
"""
Set random seeds for reproducibility.
Description: This module provides a function to set random seeds for Python, NumPy, and PyTorch.
 It also allows toggling PyTorch's deterministic operations for reproducible results at the cost of speed.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True, env_hash_seed: int | None = None) -> None:
    """
    Set all relevant RNG seeds. Optionally toggle PyTorch deterministic ops.
    """
    if env_hash_seed is not None:
        os.environ["PYTHONHASHSEED"] = str(env_hash_seed)
    else:
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # trade speed for determinism
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
