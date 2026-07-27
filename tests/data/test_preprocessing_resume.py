from pathlib import Path

from tests.data.conftest import make_subject

from fl_brats.data.indexing import build_dataset_index
from fl_brats.data.preprocessing import preprocess_index


def test_resume_detects_source_mtime_change(raw_root: Path, tmp_path: Path) -> None:
    directory = make_subject(raw_root)
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    config = {"materialize": True, "crop": {"enabled": False}, "hashing": {"source_files": False}}
    assert (
        preprocess_index(index, raw_root, tmp_path / "processed", config, resume=True)["completed"]
        == 1
    )
    path = directory / "BraTS2021_00000_t1.nii.gz"
    path.touch()
    assert (
        preprocess_index(index, raw_root, tmp_path / "processed", config, resume=True)["completed"]
        == 1
    )
