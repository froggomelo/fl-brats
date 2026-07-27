"""Dispatch layer kept explicit so SAM does not masquerade as a 3D U-Net."""

from typing import Any

import torch

from .engine import evaluate, train_local_epochs, train_one_epoch


def train_model(model: torch.nn.Module, *args: Any, **kwargs: Any) -> dict[str, float]:
    if getattr(model, "model_kind", "") == "sam":
        from .sam_engine import train_sam_epoch
        return train_sam_epoch(model, *args, **kwargs)
    return train_one_epoch(model, *args, **kwargs)


__all__ = ["evaluate", "train_local_epochs", "train_model"]
