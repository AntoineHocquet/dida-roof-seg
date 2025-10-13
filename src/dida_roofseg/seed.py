import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True, env_hash_seed: Optional[int] = None) -> None:
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
