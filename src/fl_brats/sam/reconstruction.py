"""Deterministic complete-volume reconstruction from 2D logits."""

from typing import Iterable

import torch
import torch.nn.functional as F


def reverse_resize(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def reconstruct_volume(slices: Iterable[tuple[int, torch.Tensor]], shape: tuple[int, int, int], plane: str = "axial") -> torch.Tensor:
    axis = {"sagittal": 0, "coronal": 1, "axial": 2}.get(plane)
    if axis is None:
        raise ValueError("unknown plane")
    values = list(slices)
    if not values:
        raise ValueError("cannot reconstruct an empty subject")
    channels = values[0][1].shape[0]
    volume = torch.zeros((channels, *shape), dtype=values[0][1].dtype, device=values[0][1].device)
    for index, prediction in values:
        volume.select(axis + 1, index).copy_(prediction)
    return volume
