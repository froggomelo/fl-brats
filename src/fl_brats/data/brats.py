"""Manifest-backed datasets shared by centralized and client execution."""
from pathlib import Path
from typing import Any
import csv
import torch
from monai.data import CacheDataset, Dataset, PersistentDataset  # type: ignore[attr-defined]
from torch.utils.data import DataLoader

def records_from_index(index_path: Path, subject_ids: list[str]) -> list[dict[str, Any]]:
    with index_path.open(encoding="utf-8") as f: rows = {r["subject_id"]: r for r in csv.DictReader(f)}
    return [{"image": [rows[s][f"{m}_path"] for m in ("t1", "t1ce", "t2", "flair")], "label": rows[s]["seg_path"], "subject_id": s} for s in subject_ids]

class BraTSDataset(Dataset):
    pass

def build_dataset(records: list[dict[str, Any]], transform: Any, mode: str = "dataset", cache_dir: Path | None = None) -> Dataset:
    if mode == "cache": return CacheDataset(records, transform=transform, cache_rate=1.0)
    if mode == "persistent":
        if cache_dir is None: raise ValueError("persistent dataset requires a user-owned cache directory")
        return PersistentDataset(records, transform=transform, cache_dir=cache_dir)
    return BraTSDataset(records, transform=transform)

def build_dataloader(dataset: Dataset, batch_size: int, training: bool, workers: int = 0, seed: int = 42) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=workers, generator=generator, pin_memory=torch.cuda.is_available())
