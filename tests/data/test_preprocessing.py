import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tests.data.conftest import make_subject

from fl_brats.data.indexing import build_dataset_index
from fl_brats.data.preprocessing import _validate_aligned_images, preprocess_index


def test_manifest_only_does_not_create_subject_outputs(raw_root: Path, tmp_path: Path) -> None:
    make_subject(raw_root)
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    summary = preprocess_index(index, raw_root, tmp_path / "processed", {"materialize": False})
    assert summary["mode"] == "manifest-only" and not (tmp_path / "processed" / "subjects").exists()


def test_materialization_records_transformed_geometry_and_updates_crop_affine(
    raw_root: Path, tmp_path: Path
) -> None:
    make_subject(raw_root)
    processed = tmp_path / "processed"
    index, _, _ = build_dataset_index(raw_root, processed)

    summary = preprocess_index(index, raw_root, processed, {"materialize": True})

    subject = processed / "subjects" / "BraTS2021_00000"
    image = nib.load(str(subject / "image.nii.gz"))
    segmentation = nib.load(str(subject / "segmentation.nii.gz"))
    metadata = json.loads((subject / "metadata.json").read_text())
    assert summary["completed"] == 1
    assert image.shape == (4, 4, 4, 4)
    assert segmentation.shape == (4, 4, 4)
    assert np.allclose(image.affine[:3, 3], [2, 2, 2])
    assert metadata["post_transform_geometry"]["passed"] is True
    assert metadata["segmentation_labels_present"] == [0, 1, 2, 4]


def test_materialization_can_process_subjects_in_parallel(
    raw_root: Path, tmp_path: Path
) -> None:
    make_subject(raw_root, subject_id="BraTS2021_00000")
    make_subject(raw_root, subject_id="BraTS2021_00001")
    processed = tmp_path / "processed"
    index, _, _ = build_dataset_index(raw_root, processed)

    summary = preprocess_index(
        index,
        raw_root,
        processed,
        {"materialize": True},
        workers=2,
    )

    assert summary["completed"] == 2
    assert (processed / "subjects" / "BraTS2021_00000" / "complete.json").exists()
    assert (processed / "subjects" / "BraTS2021_00001" / "complete.json").exists()


def test_materialization_optionally_resamples_all_inputs(raw_root: Path, tmp_path: Path) -> None:
    make_subject(raw_root, affine=np.diag([2.0, 2.0, 2.0, 1.0]))
    processed = tmp_path / "processed"
    index, _, _ = build_dataset_index(raw_root, processed)

    summary = preprocess_index(
        index,
        raw_root,
        processed,
        {"materialize": True, "resample": {"enabled": True, "target_spacing": [1.0, 1.0, 1.0]}},
    )

    subject = processed / "subjects" / "BraTS2021_00000"
    image = nib.load(str(subject / "image.nii.gz"))
    segmentation = nib.load(str(subject / "segmentation.nii.gz"))
    assert summary["completed"] == 1
    assert image.header.get_zooms()[:3] == (1.0, 1.0, 1.0)
    assert segmentation.header.get_zooms()[:3] == (1.0, 1.0, 1.0)
    assert set(np.unique(segmentation.get_fdata()).astype(int)) <= {0, 1, 2, 4}


def test_post_transform_geometry_check_rejects_misaligned_inputs() -> None:
    reference = nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4))
    shifted = nib.Nifti1Image(np.zeros((4, 4, 4)), np.diag([1.0, 1.0, 1.0, 1.0]))
    shifted.affine[0, 3] = 1.0

    with pytest.raises(ValueError, match="segmentation affine differs from t1"):
        _validate_aligned_images([reference, reference, reference, reference, shifted])
