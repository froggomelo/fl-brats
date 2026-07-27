"""Deterministic subject-level split manifests."""

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from .hashing import sha256_file, sha256_payload


def create_split(
    index_path: Path,
    manifest_path: Path,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, object]:
    if (
        min(train_fraction, validation_fraction, test_fraction) < 0
        or abs(train_fraction + validation_fraction + test_fraction - 1) > 1e-9
    ):
        raise ValueError("split fractions must be non-negative and sum to 1")
    rows = list(csv.DictReader(index_path.open(encoding="utf-8")))
    subjects = sorted(
        r["subject_id"]
        for r in rows
        if r.get("complete") == "True" and r.get("validation_status", "valid") == "valid"
    )
    if len(subjects) != len(set(subjects)):
        raise ValueError("index contains duplicate subject IDs")
    random.Random(seed).shuffle(subjects)
    n = len(subjects)
    first, second = int(n * train_fraction), int(n * validation_fraction)
    groups = {
        "train_subjects": subjects[:first],
        "validation_subjects": subjects[first : first + second],
        "test_subjects": subjects[first + second :],
    }
    payload: dict[str, object] = {
        "split_seed": seed,
        "requested_fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "realised_counts": {k.removesuffix("_subjects"): len(v) for k, v in groups.items()},
        "realised_fractions": {
            k.removesuffix("_subjects"): len(v) / n if n else 0 for k, v in groups.items()
        },
        "dataset_index_hash": sha256_file(index_path),
        "dataset_index_path": str(index_path.resolve()),
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        **groups,
    }
    payload["split_manifest_hash"] = sha256_payload(payload)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for key, values in groups.items():
        (manifest_path.parent / f"{key.removesuffix('_subjects')}_subjects.json").write_text(
            json.dumps(values, indent=2), encoding="utf-8"
        )
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
