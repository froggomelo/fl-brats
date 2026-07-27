from pathlib import Path

from tests.data.conftest import make_subject

from fl_brats.data.indexing import build_dataset_index
from fl_brats.data.preprocessing import preprocess_index


def test_stale_temp_is_not_completion(raw_root: Path, tmp_path: Path) -> None:
    make_subject(raw_root)
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    stale = tmp_path / "processed" / "subjects" / "BraTS2021_00000.tmp"
    stale.mkdir(parents=True)
    summary = preprocess_index(
        index, raw_root, tmp_path / "processed", {"materialize": True}, resume=True
    )
    assert (
        summary["completed"] == 1
        and (tmp_path / "processed" / "subjects" / "BraTS2021_00000" / "complete.json").exists()
    )
