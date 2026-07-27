from pathlib import Path

from tests.data.conftest import make_subject

from fl_brats.data.discovery import discover_records


def test_recursive_and_ambiguous_discovery(raw_root: Path) -> None:
    directory = make_subject(raw_root, "BraTS2021_00001")
    (directory / "BraTS2021_00001_t1_copy.nii.gz").write_bytes(b"x")
    records = discover_records(raw_root)
    assert len(records) == 1 and "t1" in records[0].duplicate_files and not records[0].complete


def test_duplicate_ids_are_retained(raw_root: Path) -> None:
    make_subject(raw_root, "BraTS2021_00001")
    make_subject(raw_root / "nested", "BraTS2021_00001")
    assert len(discover_records(raw_root)) == 2
