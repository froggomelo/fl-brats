from pathlib import Path

from tests.data.conftest import make_subject

from fl_brats.data.indexing import build_dataset_index
from fl_brats.data.splits import create_split


def test_split_is_deterministic(raw_root: Path, tmp_path: Path) -> None:
    for i in range(8):
        make_subject(raw_root, f"BraTS2021_{i:05d}")
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    assert (
        create_split(index, tmp_path / "a.json")["train_subjects"]
        == create_split(index, tmp_path / "b.json")["train_subjects"]
    )
