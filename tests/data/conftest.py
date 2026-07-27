from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    return tmp_path / "raw" / "brats2021"


def make_subject(
    root: Path,
    subject_id: str = "BraTS2021_00000",
    shape: tuple[int, int, int] = (8, 8, 8),
    affine: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> Path:
    directory = root / "CollectionA" / subject_id
    directory.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4) if affine is None else affine
    image = np.zeros(shape, dtype=np.float32)
    image[2:-2, 2:-2, 2:-2] = 5
    segmentation = np.zeros(shape, dtype=np.uint8) if labels is None else labels
    if labels is None:
        segmentation[2, 2, 2], segmentation[3, 3, 3], segmentation[4, 4, 4] = 1, 2, 4
    for role in ("t1", "t1ce", "t2", "flair"):
        nib.save(nib.Nifti1Image(image, affine), str(directory / f"{subject_id}_{role}.nii.gz"))
    nib.save(nib.Nifti1Image(segmentation, affine), str(directory / f"{subject_id}_seg.nii.gz"))
    return directory


@pytest.fixture
def complete_subject(raw_root: Path) -> Path:
    return make_subject(raw_root)
