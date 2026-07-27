"""Fixed quantity-skew partition manifests; the final client owns remainder."""

import hashlib
import json
import random
from pathlib import Path


def create_partition(split_path: Path, output_dir: Path, fractions: list[float], seed: int = 42) -> dict[str, object]:
    if not fractions or any(value <= 0 for value in fractions) or abs(sum(fractions) - 1) > 1e-9: raise ValueError("client fractions must be positive and sum to 1")
    split = json.loads(split_path.read_text(encoding="utf-8")); subjects = list(split["train_subjects"])
    if len(subjects) < len(fractions): raise ValueError("no client may be empty")
    random.Random(seed).shuffle(subjects)
    counts = [int(len(subjects) * value) for value in fractions[:-1]]; counts.append(len(subjects) - sum(counts))
    if any(value == 0 for value in counts): raise ValueError("no client may be empty")
    clients, cursor = [], 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for client_id, count in enumerate(counts):
        client_subjects = sorted(subjects[cursor:cursor + count]); cursor += count
        item = {"client_id": client_id, "subject_ids": client_subjects, "subject_count": count}
        item["manifest_hash"] = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        (output_dir / f"client_{client_id}.json").write_text(json.dumps(item, indent=2, sort_keys=True), encoding="utf-8"); clients.append(item)
    payload: dict[str, object] = {"partition_seed": seed, "requested_client_fractions": fractions, "realised_client_fractions": [count / len(subjects) for count in counts], "source_training_split_hash": split["split_manifest_hash"], "clients": clients, "rounding": "floor all but final client; final client receives remainder"}
    payload["partition_manifest_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
