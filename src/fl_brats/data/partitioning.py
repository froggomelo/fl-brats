"""Deterministic quantity-skew partition manifests with index references."""

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from .cohorts import validate_split
from .hashing import sha256_file, sha256_payload
from .modalities import (
    MODALITIES,
    MissingModalityPolicy,
    NoMissingnessPolicy,
    combination_to_mask,
    mask_to_combination_id,
    policy_fingerprint,
)


def create_partition(
    split_path: Path,
    output_dir: Path,
    fractions: list[float],
    seed: int = 42,
    index_path: Path | None = None,
    missingness_policy: MissingModalityPolicy | None = None,
    missingness_config: dict[str, object] | None = None,
) -> dict[str, object]:
    if not fractions or any(x <= 0 for x in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("client fractions must be positive and sum to 1")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validate_split(split)
    subjects = list(split["train_subjects"])
    if len(subjects) < len(fractions):
        raise ValueError("no client may be empty")
    index_path = index_path or Path(split["dataset_index_path"])
    rows = list(csv.DictReader(index_path.open(encoding="utf-8")))
    index = {r["subject_id"]: (i, r) for i, r in enumerate(rows)}
    if sha256_file(index_path) != split["dataset_index_hash"]:
        raise ValueError("dataset index differs from split manifest")
    random.Random(seed).shuffle(subjects)
    counts = [max(1, int(len(subjects) * x)) for x in fractions[:-1]]
    final = len(subjects) - sum(counts)
    while final < 1:
        donor = max(range(len(counts)), key=counts.__getitem__)
        if counts[donor] <= 1:
            raise ValueError("no client may be empty")
        counts[donor] -= 1
        final += 1
    counts.append(final)
    output_dir.mkdir(parents=True, exist_ok=True)
    clients = []
    policy = missingness_policy or NoMissingnessPolicy()
    fingerprint = policy_fingerprint(missingness_config or {"enabled": False})
    if hasattr(policy, "explicit"):
        known_clients = {f"client_{number}" for number in range(len(fractions))}
        unknown_clients = set(policy.explicit) - known_clients  # type: ignore[attr-defined]
        if unknown_clients:
            raise ValueError(
                f"missingness policy references unknown clients: {sorted(unknown_clients)}"
            )
    client_subjects, cursor = [], 0
    for count in counts:
        client_subjects.append(sorted(subjects[cursor : cursor + count]))
        cursor += count
    # Partition-level policies (dirichlet_combination) assign whole clients at once.
    assignment, proportions = policy.assign(client_subjects) if hasattr(policy, "assign") else (None, None)
    for client_id, (requested, count, ids) in enumerate(zip(fractions, counts, client_subjects, strict=True)):
        references = []
        for subject in ids:
            row_number, row = index[subject]
            available = (
                assignment[client_id][subject]
                if assignment is not None
                else policy.available_modalities(
                    subject_id=subject, client_id=f"client_{client_id}", split="train"
                )
            )
            if not available:
                raise ValueError(f"missingness policy assigned no modalities to {subject}")
            mask = combination_to_mask(available).tolist()
            references.append(
                {
                    "subject_id": subject,
                    "client_id": f"client_{client_id}",
                    "split": "train",
                    "index_row": row_number,
                    "paths": {
                        role: row[f"{role}_path"] if role in available else None
                        for role in MODALITIES
                    }
                    | {"seg": row["segmentation_path"]},
                    "available_modalities": list(available),
                    "missing_modalities": [m for m in MODALITIES if m not in available],
                    "modality_mask": mask,
                    "modality_combination_id": int(mask_to_combination_id(mask)),
                    "missingness_seed": seed,
                    "missingness_realisation_id": fingerprint,
                }
            )
        client = {
            "client_id": client_id,
            "client_name": f"client_{client_id}",
            "subject_count": count,
            "requested_fraction": requested,
            "realised_fraction": count / len(subjects),
            "subject_ids": ids,
            "subjects": references,
            "dataset_index_hash": split["dataset_index_hash"],
            "source_split_hash": split["split_manifest_hash"],
            "partition_seed": seed,
            "missingness_policy_fingerprint": fingerprint,
            "missingness_mode": (missingness_config or {}).get("mode", "none"),
        }
        client["client_manifest_hash"] = sha256_payload(client)
        (output_dir / f"client_{client_id}.json").write_text(
            json.dumps(client, indent=2, sort_keys=True), encoding="utf-8"
        )
        clients.append(client)
    payload: dict[str, object] = {
        "partition_seed": seed,
        "requested_client_fractions": fractions,
        "realised_client_fractions": [x / len(subjects) for x in counts],
        "dataset_index_hash": split["dataset_index_hash"],
        "dataset_index_path": str(index_path.resolve()),
        "source_training_split_hash": split["split_manifest_hash"],
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "rounding": (
            "floor all but final client; final client receives remainder "
            "(minimum one subject where feasible)"
        ),
        "clients": clients,
        "missingness_policy_fingerprint": fingerprint,
    }
    if proportions is not None:
        payload["client_combination_proportions"] = proportions
    payload["partition_manifest_hash"] = sha256_payload(payload)
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload
