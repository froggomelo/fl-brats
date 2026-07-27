"""Deterministic, subject-level data split manifests."""

import hashlib
import json
import random
from pathlib import Path

from .indexing import sha256_file


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def create_split(index_path: Path, manifest_path: Path, seed: int = 42, train_fraction: float = .70, validation_fraction: float = .15, test_fraction: float = .15) -> dict[str, object]:
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    import csv
    with index_path.open(encoding="utf-8") as f: rows = list(csv.DictReader(f))
    subjects = sorted(row["subject_id"] for row in rows if row.get("complete") == "True")
    random.Random(seed).shuffle(subjects)
    n, train_n, valid_n = len(subjects), int(len(subjects) * train_fraction), int(len(subjects) * validation_fraction)
    payload: dict[str, object] = {"split_seed": seed, "source_dataset_index_hash": sha256_file(index_path), "train_subjects": subjects[:train_n], "validation_subjects": subjects[train_n:train_n + valid_n], "test_subjects": subjects[train_n + valid_n:], "fractions": {"train": train_fraction, "validation": validation_fraction, "test": test_fraction}}
    payload["split_manifest_hash"] = _hash(payload)
    manifest_path.parent.mkdir(parents=True, exist_ok=True); manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
