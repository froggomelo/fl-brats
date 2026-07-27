from pathlib import Path

from tests.data.conftest import make_subject

from fl_brats.data.indexing import build_dataset_index


def test_index_paths_are_relative(raw_root: Path, tmp_path: Path) -> None:
    make_subject(raw_root)
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    assert "raw" not in index.read_text().splitlines()[1].split(",")[3]
