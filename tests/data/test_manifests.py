from pathlib import Path

from tests.data.conftest import make_subject

from fl_brats.data.indexing import build_dataset_index
from fl_brats.data.manifests import verify_manifests
from fl_brats.data.partitioning import create_partition
from fl_brats.data.splits import create_split


def test_manifest_verification(raw_root: Path, tmp_path: Path) -> None:
    for i in range(8):
        make_subject(raw_root, f"BraTS2021_{i:05d}")
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    root = tmp_path / "partitions"
    create_split(index, root / "split_manifest.json")
    create_partition(root / "split_manifest.json", root, [0.5, 0.25, 0.15, 0.1])
    assert verify_manifests(root) == []
