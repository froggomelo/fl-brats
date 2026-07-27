"""SAM slice training entry point.

Callers provide slice loaders. Subject evaluation must reconstruct all slices via
``fl_brats.sam.reconstruction`` before computing Dice.
"""

from typing import Any

import torch

from .engine import train_one_epoch


def train_sam_epoch(model: torch.nn.Module, loader: Any, loss_fn: Any, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    return train_one_epoch(model, loader, loss_fn, optimizer, device)
