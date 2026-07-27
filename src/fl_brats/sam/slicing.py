"""Deterministic 3D-to-2D extraction retaining reconstruction metadata."""

from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class SliceRecord:
    subject_id: str
    slice_index: int
    plane: str
    original_shape: tuple[int, int, int]
    tumour_present: bool


def _axis(plane: str) -> int:
    return {"sagittal": 0, "coronal": 1, "axial": 2}.get(plane, -1)


def extract_slices(image: torch.Tensor, label: torch.Tensor | None = None, subject_id: str = "", plane: str = "axial") -> Iterator[tuple[torch.Tensor, torch.Tensor | None, SliceRecord]]:
    """Image has [4,D,H,W], label has [3,D,H,W]; axial is the default."""
    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError("expected [4,D,H,W]")
    axis = _axis(plane)
    if axis < 0:
        raise ValueError("plane must be axial, coronal, or sagittal")
    spatial = tuple(image.shape[1:])
    for index in range(spatial[axis]):
        img = image.select(axis + 1, index)
        target = label.select(axis + 1, index) if label is not None else None
        present = bool(target.any()) if target is not None else False
        yield img, target, SliceRecord(subject_id, index, plane, spatial, present)


def select_training_slices(records: list[SliceRecord], negative_ratio: float | None = None) -> list[SliceRecord]:
    positives = [x for x in records if x.tumour_present]
    negatives = [x for x in records if not x.tumour_present]
    if negative_ratio is None:
        return records
    return positives + negatives[: int(round(len(positives) * negative_ratio))]
