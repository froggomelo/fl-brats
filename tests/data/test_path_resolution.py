from pathlib import Path

from pytest import MonkeyPatch

from tests.data.conftest import make_subject

from fl_brats.data.brats import records_from_client_manifest, records_from_index
from fl_brats.data.indexing import build_dataset_index
from fl_brats.data.partitioning import create_partition
from fl_brats.data.splits import create_split


def test_index_paths_are_relative(raw_root: Path, tmp_path: Path) -> None:
    make_subject(raw_root)
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    assert "raw" not in index.read_text().splitlines()[1].split(",")[3]


def test_records_from_index_resolves_relative_paths(raw_root: Path, tmp_path: Path) -> None:
    subject_dir = make_subject(raw_root)
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    record = records_from_index(index, ["BraTS2021_00000"])[0]
    assert record["image"][0] == str(subject_dir / "BraTS2021_00000_t1.nii.gz")
    assert record["label"] == str(subject_dir / "BraTS2021_00000_seg.nii.gz")


def test_records_from_index_prefers_raw_root_env(
    raw_root: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    make_subject(raw_root)
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    override = tmp_path / "relocated"
    monkeypatch.setenv("FL_BRATS_RAW_ROOT", str(override))
    record = records_from_index(index, ["BraTS2021_00000"])[0]
    assert record["image"][0] == str(
        override / "BraTS2021_00000" / "BraTS2021_00000_t1.nii.gz"
    )


def test_records_from_client_manifest_resolves_relative_paths(
    raw_root: Path, tmp_path: Path
) -> None:
    for i in range(8):
        make_subject(raw_root, f"BraTS2021_{i:05d}")
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    split = tmp_path / "partitions" / "split_manifest.json"
    create_split(index, split)
    create_partition(split, split.parent, [0.5, 0.25, 0.15, 0.1])

    record = records_from_client_manifest(split.parent / "client_0.json")[0]

    assert str(raw_root) in record["image"][0]
    assert str(raw_root) in record["label"]


def test_records_from_client_manifest_prefers_raw_root_env(
    raw_root: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    for i in range(8):
        make_subject(raw_root, f"BraTS2021_{i:05d}")
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    split = tmp_path / "partitions" / "split_manifest.json"
    create_split(index, split)
    create_partition(split, split.parent, [0.5, 0.25, 0.15, 0.1])
    override = tmp_path / "relocated"
    monkeypatch.setenv("FL_BRATS_RAW_ROOT", str(override))

    record = records_from_client_manifest(split.parent / "client_0.json")[0]

    assert str(override) in record["image"][0]
    assert str(override) in record["label"]
