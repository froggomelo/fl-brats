"""Explicit four-modality MRI to three-channel SAM image conversion."""

from typing import Sequence

import torch
from torch import nn

CANONICAL_MODALITIES = ("t1", "t1ce", "t2", "flair")


class LearnedFourToThree(nn.Module):
    """Federated trainable 1x1 projection; no labels are involved."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 4:
            raise ValueError("expected [B, 4, H, W] MRI slices")
        return self.projection(image)


def fixed_rgb_mapping(image: torch.Tensor, mapping: Sequence[str] = ("flair", "t1ce", "t2")) -> torch.Tensor:
    """Return configured channels, explicitly documenting omitted modalities."""
    if image.ndim != 4 or image.shape[1] != 4:
        raise ValueError("expected [B, 4, H, W] MRI slices")
    unknown = set(mapping).difference(CANONICAL_MODALITIES)
    if len(mapping) != 3 or unknown:
        raise ValueError("mapping must contain exactly three canonical modalities")
    indexes = [CANONICAL_MODALITIES.index(name) for name in mapping]
    return image[:, indexes]
