"""Subject boundaries shared by training, validation, and final reporting."""

import json
from pathlib import Path
from typing import Any


def validate_split(split: dict[str, Any]) -> None:
    groups = {}
    for name in ("train", "validation", "test"):
        ids = split[f"{name}_subjects"]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {name} subjects")
        groups[name] = set(ids)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups[left] & groups[right]:
            raise ValueError(f"split overlap: {left}/{right}")


def split_path_for(values: dict[str, Any]) -> Path:
    data = values.get("data", {})
    configured = values.get("split_path") or data.get("split_path")
    if configured and Path(configured).is_file():
        return Path(configured)
    partition = values.get("partition_root") or data.get("partition_root")
    frozen = Path(partition) / "split_manifest.json" if partition else None
    if frozen and frozen.is_file():
        return frozen
    raise ValueError(
        "held-out evaluation requires data.split_path or partition/split_manifest.json"
    )


def read_split(path: Path) -> dict[str, Any]:
    split = json.loads(path.read_text(encoding="utf-8"))
    validate_split(split)
    return split


def validate_client_manifest(path: Path, split: dict[str, Any]) -> list[str]:
    """Inspect IDs only; the evaluation process never pools client images."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = [item["subject_id"] for item in payload["subjects"]]
    if len(ids) != len(set(ids)) or not set(ids) <= set(split["train_subjects"]):
        raise ValueError(f"client manifest must contain unique training subjects only: {path}")
    if any(item.get("split", "train") != "train" for item in payload["subjects"]):
        raise ValueError(f"non-training reference in {path}")
    if "subject_ids" in payload and sorted(payload["subject_ids"]) != sorted(ids):
        raise ValueError(f"subject_ids disagree with references in {path}")
    return ids


def validate_client_missingness(path: Path, missing_config: dict[str, Any] | None) -> None:
    """Reject a partition whose baked-in modality masks disagree with the config.

    Masks are fixed when the partition is created, so pointing a config at a
    partition built with another missingness policy would otherwise train
    silently on the wrong availability pattern.
    """
    from fl_brats.data.modalities import MODALITIES, canonical_modality_order, policy_from_config

    policy = policy_from_config(missing_config or {})
    payload = json.loads(path.read_text(encoding="utf-8"))
    client_id = payload.get("client_name") or f"client_{payload.get('client_id', path.stem)}"
    if hasattr(policy, "assign"):
        # Partition-level policy: re-derive every client's assignment from the
        # seed and the partition's subject lists, then compare this client.
        clients = sorted(
            (json.loads(other.read_text(encoding="utf-8")) for other in path.parent.glob("client_*.json")),
            key=lambda client: int(client["client_id"]),
        )
        assignment, _ = policy.assign([sorted(client["subject_ids"]) for client in clients])
        expected = assignment[[int(client["client_id"]) for client in clients].index(int(payload["client_id"]))]
    mismatched = [
        item["subject_id"]
        for item in payload["subjects"]
        if canonical_modality_order(item.get("available_modalities", list(MODALITIES)))
        != (
            canonical_modality_order(expected[item["subject_id"]])
            if hasattr(policy, "assign")
            else policy.available_modalities(
                subject_id=item["subject_id"], client_id=client_id, split="train"
            )
        )
    ]
    if mismatched:
        raise ValueError(
            f"{path}: modality masks of {len(mismatched)}/{len(payload['subjects'])} subjects "
            "differ from the configured missing_modalities policy; recreate the partition "
            "with prepare-experiment-partition for this config"
        )
