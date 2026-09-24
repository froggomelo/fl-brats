"""Manifest-backed datasets shared by centralized and client execution."""

import csv
import hashlib
import json
import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from monai.data import (  # type: ignore[attr-defined]
    CacheDataset,
    Dataset,
    PersistentDataset,
    list_data_collate,
)
from monai.data.utils import worker_init_fn as monai_worker_init_fn
from monai.utils import convert_to_tensor
from torch.utils.data import DataLoader

from .modalities import MODALITIES, combination_to_mask, mask_to_combination_id


def _env_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _environment_raw_root() -> Path | None:
    """Return the canonical raw-data root configured for this process."""
    if raw_root := os.environ.get("FL_BRATS_RAW_ROOT"):
        return Path(raw_root)
    if data_root := os.environ.get("FL_BRATS_DATA_ROOT"):
        return Path(data_root) / "raw" / "brats2021"
    return None


def _index_path_root(index_path: Path) -> Path | None:
    if raw_root := _environment_raw_root():
        return raw_root
    summary_path = index_path.with_name("dataset_summary.json")
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_root = summary.get("raw_root")
    return Path(raw_root) if raw_root else None


def _resolve_record_path(path: str, root: Path | None) -> str:
    value = Path(path)
    if value.is_absolute() or not path:
        return path
    if root is None:
        raise RuntimeError(
            "cannot resolve relative BraTS manifest path because no dataset root is "
            "configured; set FL_BRATS_RAW_ROOT or FL_BRATS_DATA_ROOT"
        )
    return str(root / value)


def _client_manifest_root(path: Path) -> Path | None:
    if raw_root := _environment_raw_root():
        return raw_root
    partition_path = path.with_name("manifest.json")
    if not partition_path.exists():
        return None
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    index_value = partition.get("dataset_index_path")
    if not index_value:
        return None
    return _index_path_root(Path(index_value))


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
            "label": _resolve_record_path(rows[s].get("seg_path") or rows[s]["segmentation_path"], root),
            "subject_id": s,
            "client_id": "centralized",
            "available_modalities": MODALITIES,
            "modality_mask": combination_to_mask(MODALITIES),
        }
        for s in subject_ids
    ]


def _attach_metadata(result: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    mask = (
        record["modality_mask"].clone()
        if isinstance(record["modality_mask"], torch.Tensor)
        else torch.tensor(record["modality_mask"], dtype=torch.bool)
    )
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


def _with_metadata(
    result: dict[str, Any] | list[dict[str, Any]], record: dict[str, Any]
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(result, list):
        return [_attach_metadata(sample, record) for sample in result]
    return _attach_metadata(result, record)


class BraTSDataset(Dataset):
    """MONAI Dataset retaining explicit availability metadata."""
    def __init__(self, data: list[dict[str, Any]], transform: Any = None) -> None:
        super().__init__(data=data, transform=transform)

    def __getitem__(self, index: int) -> dict[str, Any] | list[dict[str, Any]]:
        return _with_metadata(super().__getitem__(index), self.data[index])


def _source_file_hash(record: dict[str, Any]) -> bytes:
    """Key cached preprocessing on the files it reads, not on client or mask metadata.

    The deterministic transforms only see the loaded image and label files, so a
    subject with the same available modalities shares one cache entry across
    clients, missingness levels, and experiments.
    """
    payload = json.dumps({"image": list(record["image"]), "label": record["label"]})
    return hashlib.sha256(payload.encode()).hexdigest().encode()


class BraTSPersistentDataset(PersistentDataset):
    """Disk-cache the deterministic transform prefix; random crops and flips stay live."""
    def __init__(
        self, data: list[dict[str, Any]], transform: Any, cache_dir: Path
    ) -> None:
        super().__init__(
            data=data, transform=transform, cache_dir=cache_dir, hash_func=_source_file_hash
        )

    def __getitem__(self, index: int) -> dict[str, Any] | list[dict[str, Any]]:  # type: ignore[override]
        return _with_metadata(super().__getitem__(index), self.data[index])

    def _cachecheck(self, item_transformed: Any) -> Any:
        """Read or build one cache entry safely when many nodes share the cache.

        MONAI writes to a temporary directory on /tmp and then moves the file,
        which across filesystems is a copy: clients on other nodes could read a
        half-written file, and a job killed mid-copy left a truncated one that
        MONAI re-raises on. Write beside the target and rename atomically, and
        rebuild any entry that fails to load.
        """
        hashfile = self.cache_dir / f"{self.hash_func(item_transformed).decode('utf-8')}.pt"
        if hashfile.is_file():
            try:
                return torch.load(hashfile, weights_only=self.weights_only)
            except Exception:  # truncated or concurrently replaced entry: rebuild it
                pass
        transformed = self._pre_transform(deepcopy(item_transformed))
        temporary = hashfile.with_name(f".{hashfile.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            # Crops return views; torch.save writes a view's whole storage, so
            # clone to store only the cropped volume (entries were ~2.7x larger).
            payload = convert_to_tensor(transformed, convert_numeric=False, track_meta=self.track_meta)
            if isinstance(payload, dict):
                payload = {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in payload.items()
                }
            torch.save(payload, temporary)
            os.replace(temporary, hashfile)
        except OSError:
            temporary.unlink(missing_ok=True)
        return transformed


def persistent_cache_dir(data: dict[str, Any]) -> Path:
    """Cache directory keyed on every setting that shapes the cached tensors."""
    settings = {
        "version": "brats-deterministic-v2",
        "orientation": data.get("orientation", "RAS"),
        "spacing": data.get("spacing", [1.0, 1.0, 1.0]),
        "patch_size": data.get("patch_size", [96, 96, 96]),
        "divisible_by": data.get("divisible_by", 16),
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    if data.get("cache_dir"):
        root = Path(data["cache_dir"])
    elif data.get("cache_location", "shared") == "node":
        # An uncompressed cached sample is ~7x its .nii.gz source, so a cache on
        # NFS trades CPU for network I/O that slow nodes cannot sustain. Cluster
        # nodes have local SSDs; /var/tmp lives there and Apptainer binds it.
        user = os.environ.get("USER") or "user"
        root = Path(os.environ.get("FL_BRATS_NODE_CACHE_ROOT", f"/var/tmp/{user}/fl-brats-cache"))
    else:
        root = Path(os.environ.get("FL_BRATS_CACHE_ROOT", "cache")) / "persistent"
    return root / digest


def records_from_client_manifest(path: Path) -> list[dict[str, Any]]:
    """Load client records without ever exposing unavailable modality paths to MONAI."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = _client_manifest_root(path)
    records = []
    for reference in payload["subjects"]:
        mask = torch.tensor(reference.get("modality_mask", [True] * 4), dtype=torch.bool)
        if not mask.any():
            raise ValueError(f"subject {reference['subject_id']} has no available modalities")
        # MONAI LoadImaged accepts a list of files, so use a zero tensor later only for missing slots.
        records.append(
            {
                "image": [
                    _resolve_record_path(reference["paths"][m], root)
                    for m in MODALITIES
                    if mask[MODALITIES.index(m)]
                ],
                "label": _resolve_record_path(reference["paths"]["seg"], root),
                "subject_id": reference["subject_id"],
                "client_id": str(payload["client_id"]),
                "available_modalities": tuple(
                    reference.get("available_modalities", MODALITIES)
                ),
                "modality_mask": mask,
            }
        )
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
        return BraTSPersistentDataset(records, transform=transform, cache_dir=cache_dir)
    return BraTSDataset(records, transform=transform)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    training: bool,
    workers: int = 0,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    if pin_memory is None:
        pin_memory = _env_flag("FL_BRATS_PIN_MEMORY")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    # Each worker process starts with a copy of the transforms' random state.
    # MONAI's worker_init_fn reseeds them per worker from the loader generator;
    # without it every worker would produce the same crops and flips.
    # Persistent workers keep those streams advancing across epochs and rounds.
    # Spawn, not fork: Flower's gRPC threads are live when workers start, and
    # forked children of a gRPC process abort ("skipping fork() handlers").
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        generator=generator,
        pin_memory=pin_memory,
        collate_fn=list_data_collate,
        worker_init_fn=monai_worker_init_fn if workers > 0 else None,
        persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers > 0 else None,
    )
