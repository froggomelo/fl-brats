"""Lightweight, non-mutating NIfTI inspection."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class NiftiInfo:
    shape: tuple[int, ...]
    spacing: tuple[float, float, float]
    orientation: str
    dtype: str
    affine: list[list[float]]


def inspect_nifti(path: Path) -> NiftiInfo:
    if not path.is_file():
        raise ValueError("file does not exist")
    if path.stat().st_size == 0:
        raise ValueError("zero-byte file")
    image: Any = nib.load(str(path))
    if image.ndim != 3:
        raise ValueError(f"expected 3D image, got {image.ndim}D")
    affine = np.asarray(image.affine, dtype=float)
    data = np.asanyarray(image.dataobj)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("invalid affine")
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    if len(spacing) != 3 or not np.isfinite(spacing).all() or any(v <= 0 for v in spacing):
        raise ValueError("invalid voxel spacing")
    if data.size == 0:
        raise ValueError("empty image")
    if not np.isfinite(data).all():
        raise ValueError("non-finite image data")
    return NiftiInfo(
        tuple(int(v) for v in image.shape),
        spacing,
        "".join(nib.aff2axcodes(affine)),  # type: ignore[no-untyped-call]
        str(data.dtype),
        affine.tolist(),
    )


def nifti_info_dict(info: NiftiInfo) -> dict[str, object]:
    return asdict(info)
