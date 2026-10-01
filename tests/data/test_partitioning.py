from pathlib import Path

from tests.data.conftest import make_subject

import json

import pytest

from fl_brats.data.cohorts import validate_client_missingness
from fl_brats.data.indexing import build_dataset_index
from fl_brats.data.modalities import policy_from_config
from fl_brats.data.partitioning import create_partition
from fl_brats.data.splits import create_split


def test_client_manifest_has_paths(raw_root: Path, tmp_path: Path) -> None:
    for i in range(8):
        make_subject(raw_root, f"BraTS2021_{i:05d}")
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    split = tmp_path / "partitions" / "split_manifest.json"
    create_split(index, split)
    payload = create_partition(split, split.parent, [0.5, 0.25, 0.15, 0.1])
    assert payload["clients"][0]["subjects"][0]["paths"]["seg"].endswith("_seg.nii.gz")  # type: ignore[index]


def test_dirichlet_combination_partition_writes_and_validates_masks(raw_root: Path, tmp_path: Path) -> None:
    for i in range(50):
        make_subject(raw_root, f"BraTS2021_{i:05d}")
    index, _, _ = build_dataset_index(raw_root, tmp_path / "processed")
    split = tmp_path / "partitions" / "split_manifest.json"
    create_split(index, split)
    config = {"enabled": True, "mode": "dirichlet_combination", "seed": 42, "dirichlet_combination": {"alpha": 0.1}}
    payload = create_partition(split, split.parent, [0.5, 0.5], missingness_policy=policy_from_config(config), missingness_config=config)
    assert len(payload["client_combination_proportions"]) == 2  # type: ignore[arg-type]
    for path in sorted(split.parent.glob("client_*.json")):
        validate_client_missingness(path, config)
        client = json.loads(path.read_text(encoding="utf-8"))
        assert len({item["modality_combination_id"] for item in client["subjects"]}) == 15
    # A tampered mask is caught by re-deriving the assignment from the seed.
    path = split.parent / "client_0.json"
    client = json.loads(path.read_text(encoding="utf-8"))
    first = client["subjects"][0]
    first["available_modalities"] = ["t1"] if first["available_modalities"] != ["t1"] else ["t2"]
    path.write_text(json.dumps(client), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from the configured"):
        validate_client_missingness(path, config)
