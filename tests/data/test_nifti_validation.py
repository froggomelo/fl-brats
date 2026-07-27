from pathlib import Path

from tests.data.conftest import make_subject

from fl_brats.data.discovery import discover_records
from fl_brats.data.validation import validate_subject


def test_zero_byte_and_shape_failures_accumulate(raw_root: Path) -> None:
    directory = make_subject(raw_root)
    (directory / "BraTS2021_00000_t1.nii.gz").write_bytes(b"")
    result = validate_subject(discover_records(raw_root)[0])
    assert not result.valid and result.unreadable_files[0]["role"] == "t1"
