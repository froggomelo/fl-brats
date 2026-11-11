import shutil
from pathlib import Path

from fl_brats.data.indexing import build_dataset_index
from fl_brats.data.manifests import verify_manifests
from fl_brats.data.partitioning import create_partition
from fl_brats.data.splits import create_split
from tests.data.conftest import make_subject


def test_manifest_verification(raw_root: Path, tmp_path: Path) -> None:
    for i in range(8):
        make_subject(raw_root, f"BraTS2021_{i:05d}")
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    root = tmp_path / "partitions"
    create_split(index, root / "split_manifest.json")
    create_partition(root / "split_manifest.json", root, [0.5, 0.25, 0.15, 0.1])
    assert verify_manifests(root) == []


def test_manifest_verification_with_external_split_and_moved_index(
    raw_root: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FL_BRATS_RAW_ROOT", str(raw_root))
    for i in range(8):
        make_subject(raw_root, f"BraTS2021_{i:05d}")
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    split = tmp_path / "splits" / "split.json"
    create_split(index, split)
    root = tmp_path / "partitions"
    create_partition(split, root, [0.5, 0.25, 0.15, 0.1])
    (root / "split_manifest.json").unlink(missing_ok=True)
    moved = tmp_path / "published" / "dataset_index.csv"
    moved.parent.mkdir()
    shutil.copy2(index, moved)
    index.unlink()
    assert verify_manifests(root) == ["missing split or partition manifest (pass --split-path)"]
    assert verify_manifests(root, split_path=split, index_path=moved) == []
