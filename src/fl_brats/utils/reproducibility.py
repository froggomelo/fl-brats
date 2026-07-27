"""Deterministic execution helpers."""

import os
import random
from collections.abc import Callable

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> Callable[[int], None]:
    """Seed Python, NumPy and PyTorch, returning a future worker seed function.

    ``PYTHONHASHSEED`` should also be set before interpreter startup; setting it
    here cannot retroactively change hashing in this interpreter.
    """
    if seed < 0:
        raise ValueError("seed must be zero or greater")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    def seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return seed_worker
