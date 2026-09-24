from pathlib import Path

import torch

from tests.data.conftest import make_subject

from fl_brats.data.brats import (
    BraTSDataset,
    BraTSPersistentDataset,
    persistent_cache_dir,
)
from fl_brats.data.modalities import combination_to_mask
from fl_brats.data.transforms import build_transforms


def _record(directory: Path, subject: str, available: tuple[str, ...]) -> dict:
    return {
        "image": [str(directory / f"{subject}_{m}.nii.gz") for m in available],
        "label": str(directory / f"{subject}_seg.nii.gz"),
        "subject_id": subject,
        "client_id": "0",
        "available_modalities": available,
        "modality_mask": combination_to_mask(available),
    }


def test_persistent_cache_matches_uncached_and_restores_missing_channels(
    raw_root: Path, tmp_path: Path
) -> None:
    directory = make_subject(raw_root, shape=(20, 20, 20))
    records = [
        _record(directory, "BraTS2021_00000", ("t1", "t1ce", "t2", "flair")),
        _record(directory, "BraTS2021_00000", ("t1", "flair")),
    ]
    data = {"patch_size": [16, 16, 16], "divisible_by": 4, "cache_dir": str(tmp_path / "c")}
    transform = build_transforms(data, False)
    cached = BraTSPersistentDataset(records, transform, persistent_cache_dir(data))
    plain = BraTSDataset(records, transform)
    for index in range(len(records)):
        first, second, expected = cached[index], cached[index], plain[index]
        for sample in (first, second):
            assert torch.equal(torch.as_tensor(sample["image"]), torch.as_tensor(expected["image"]))
            assert torch.equal(torch.as_tensor(sample["label"]), torch.as_tensor(expected["label"]))
            assert torch.equal(sample["modality_mask"], expected["modality_mask"])
    assert cached[1]["image"].shape[0] == 4
    assert not cached[1]["image"][1].any() and not cached[1]["image"][2].any()
    assert len(list(persistent_cache_dir(data).glob("*.pt"))) == 2


def test_cache_dir_changes_with_preprocessing_settings(tmp_path: Path) -> None:
    base = {"cache_dir": str(tmp_path)}
    assert persistent_cache_dir(base) == persistent_cache_dir(dict(base))
    assert persistent_cache_dir(base) != persistent_cache_dir({**base, "patch_size": [64, 64, 64]})


def test_truncated_cache_entry_is_rebuilt(raw_root: Path, tmp_path: Path) -> None:
    directory = make_subject(raw_root, shape=(20, 20, 20))
    records = [_record(directory, "BraTS2021_00000", ("t1", "t1ce", "t2", "flair"))]
    data = {"patch_size": [16, 16, 16], "divisible_by": 4, "cache_dir": str(tmp_path / "c")}
    transform = build_transforms(data, False)
    expected = BraTSDataset(records, transform)[0]
    cache_dir = persistent_cache_dir(data)
    BraTSPersistentDataset(records, transform, cache_dir)[0]
    (entry,) = cache_dir.glob("*.pt")
    entry.write_bytes(entry.read_bytes()[:100])
    rebuilt = BraTSPersistentDataset(records, transform, cache_dir)[0]
    assert torch.equal(torch.as_tensor(rebuilt["image"]), torch.as_tensor(expected["image"]))
    assert len(entry.read_bytes()) > 100
    assert not list(cache_dir.glob(".*.tmp"))


def test_node_cache_location_uses_local_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FL_BRATS_NODE_CACHE_ROOT", str(tmp_path / "node"))
    monkeypatch.setenv("FL_BRATS_CACHE_ROOT", str(tmp_path / "shared"))
    assert persistent_cache_dir({"cache_location": "node"}).parent == tmp_path / "node"
    assert persistent_cache_dir({}).parent == tmp_path / "shared" / "persistent"


def test_cache_holds_full_deterministic_prefix_without_view_storage(
    raw_root: Path, tmp_path: Path
) -> None:
    directory = make_subject(raw_root, shape=(24, 24, 24))
    records = [_record(directory, "BraTS2021_00000", ("t1", "t1ce", "t2", "flair"))]
    data = {"patch_size": [8, 8, 8], "divisible_by": 4, "cache_dir": str(tmp_path / "c")}
    BraTSPersistentDataset(records, build_transforms(data, True), persistent_cache_dir(data))[0]
    (entry,) = persistent_cache_dir(data).glob("*.pt")
    cached = torch.load(entry, weights_only=False)
    # Region conversion and padding are cached, so the label has WT/TC/ET channels.
    assert cached["label"].shape[0] == 3
    assert all(size % 4 == 0 for size in cached["image"].shape[1:])
    # Only the cropped volume is stored, not the uncropped storage behind a view.
    stored = sum(
        value.untyped_storage().nbytes()
        for value in cached.values()
        if isinstance(value, torch.Tensor)
    )
    tensors = sum(
        value.numel() * value.element_size()
        for value in cached.values()
        if isinstance(value, torch.Tensor)
    )
    assert stored == tensors
