from pathlib import Path

from tests.data.conftest import make_subject

from fl_brats.data.indexing import build_dataset_index


def test_index_reports_unreadable_detail(raw_root: Path, tmp_path: Path) -> None:
    directory = make_subject(raw_root)
    (directory / "BraTS2021_00000_t2.nii.gz").write_bytes(b"")
    _, _, summary = build_dataset_index(raw_root, tmp_path / "processed")
    assert summary["unreadable_file_count"] == 1 and summary["unreadable_files_by_role"] == {
        "t2": 1
    }
