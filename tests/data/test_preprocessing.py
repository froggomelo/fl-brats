from pathlib import Path

from tests.data.conftest import make_subject

from fl_brats.data.indexing import build_dataset_index
from fl_brats.data.preprocessing import preprocess_index


def test_manifest_only_does_not_create_subject_outputs(raw_root: Path, tmp_path: Path) -> None:
    make_subject(raw_root)
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    summary = preprocess_index(index, raw_root, tmp_path / "processed", {"materialize": False})
    assert summary["mode"] == "manifest-only" and not (tmp_path / "processed" / "subjects").exists()
