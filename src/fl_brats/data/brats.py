"""Manifest-backed datasets shared by centralized and client execution."""

import csv
import json
from pathlib import Path
from typing import Any

import torch
from monai.data import CacheDataset, Dataset, PersistentDataset  # type: ignore[attr-defined]
from torch.utils.data import DataLoader

from .modalities import MODALITIES, combination_to_mask, mask_to_combination_id


def _index_path_root(index_path: Path) -> Path | None:
    summary_path = index_path.with_name("dataset_summary.json")
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_root = summary.get("raw_root")
    return Path(raw_root) if raw_root else None


def _resolve_record_path(path: str, root: Path | None) -> str:
    value = Path(path)
    if value.is_absolute() or root is None or not path:
        return path
    return str(root / value)


def records_from_index(index_path: Path, subject_ids: list[str]) -> list[dict[str, Any]]:
    root = _index_path_root(index_path)
    with index_path.open(encoding="utf-8") as f:
        rows = {r["subject_id"]: r for r in csv.DictReader(f)}
    return [
        {
            "image": [
                _resolve_record_path(rows[s][f"{m}_path"], root)
                for m in ("t1", "t1ce", "t2", "flair")
            ],
            "label": _resolve_record_path(rows[s]["seg_path"], root),
            "subject_id": s,
            "client_id": "centralized",
            "available_modalities": MODALITIES,
            "modality_mask": combination_to_mask(MODALITIES),
        }
        for s in subject_ids
    ]


class BraTSDataset(Dataset):
    """MONAI Dataset retaining explicit availability metadata."""
    def __init__(self, data: list[dict[str, Any]], transform: Any = None) -> None:
        super().__init__(data=data, transform=transform)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.data[index]
        result = super().__getitem__(index)
        mask = record["modality_mask"].clone() if isinstance(record["modality_mask"], torch.Tensor) else torch.tensor(record["modality_mask"], dtype=torch.bool)
        image = result["image"]
        # LoadImaged stacks only available images. Restore the invariant four-channel layout.
        if image.shape[0] != 4:
            restored = torch.zeros((4, *image.shape[1:]), dtype=image.dtype, device=image.device)
            restored[mask] = image
            result["image"] = restored
        result["modality_mask"] = mask
        result["modality_combination_id"] = mask_to_combination_id(mask)
        result["available_modalities"] = tuple(record["available_modalities"])
        result["subject_id"] = record["subject_id"]
        result["client_id"] = record["client_id"]
        return result


def records_from_client_manifest(path: Path) -> list[dict[str, Any]]:
    """Load client records without ever exposing unavailable modality paths to MONAI."""
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for reference in payload["subjects"]:
        mask = torch.tensor(reference.get("modality_mask", [True] * 4), dtype=torch.bool)
        if not mask.any():
            raise ValueError(f"subject {reference['subject_id']} has no available modalities")
        # MONAI LoadImaged accepts a list of files, so use a zero tensor later only for missing slots.
        records.append({"image": [reference["paths"][m] for m in MODALITIES if mask[MODALITIES.index(m)]],
                        "label": reference["paths"]["seg"], "subject_id": reference["subject_id"],
                        "client_id": str(payload["client_id"]), "available_modalities": tuple(reference.get("available_modalities", MODALITIES)),
                        "modality_mask": mask})
    return records


def build_dataset(
    records: list[dict[str, Any]],
    transform: Any,
    mode: str = "dataset",
    cache_dir: Path | None = None,
) -> Dataset:
    if mode == "cache":
        return CacheDataset(records, transform=transform, cache_rate=1.0)
    if mode == "persistent":
        if cache_dir is None:
            raise ValueError("persistent dataset requires a user-owned cache directory")
        return PersistentDataset(records, transform=transform, cache_dir=cache_dir)
    return BraTSDataset(records, transform=transform)


def build_dataloader(
    dataset: Dataset, batch_size: int, training: bool, workers: int = 0, seed: int = 42
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
