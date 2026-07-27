"""Controlled baseline comparison reporting."""

import csv
import json
from pathlib import Path
from typing import Any

CONTROL_KEYS = (
    "dataset_index_hash",
    "split_manifest_hash",
    "partition_manifest_hash",
    "initial_weight_hash",
    "model_config",
    "loss_config",
    "preprocessing_config",
    "evaluation_policy",
)


def compare(
    centralized_path: Path, federated_path: Path, output_dir: Path, strict: bool = True
) -> dict[str, Any]:
    c, f = json.loads(centralized_path.read_text()), json.loads(federated_path.read_text())
    mismatches = [key for key in CONTROL_KEYS if c.get(key) != f.get(key)]
    if mismatches and strict:
        raise ValueError(f"experiments do not match controls: {', '.join(mismatches)}")
    row: dict[str, Any] = {"control_mismatches": mismatches}
    for region in ("wt", "tc", "et", "mean"):
        key = f"test_{region}_dice"
        row[f"centralized_{region}_dice"] = c.get(key)
        row[f"federated_{region}_dice"] = f.get(key)
        row[f"{region}_dice_difference"] = c.get(key, 0) - f.get(key, 0)
    row.update(
        {
            "centralized_training_time": c.get("total_training_time"),
            "federated_wall_clock_time": f.get("total_wall_clock_time"),
            "communication_rounds": f.get("communication_rounds"),
            "local_epochs": f.get("local_epochs"),
            "cumulative_communication_bytes": f.get("cumulative_model_payload_bytes"),
            "number_of_clients": f.get("number_of_clients"),
            "model_parameter_count": c.get("model_parameter_count"),
            "training_subject_count": c.get("training_subject_count"),
            "initial_weight_hash_match": c.get("initial_weight_hash")
            == f.get("initial_weight_hash"),
            "split_manifest_hash_match": c.get("split_manifest_hash")
            == f.get("split_manifest_hash"),
            "partition_manifest_hash_match": c.get("partition_manifest_hash")
            == f.get("partition_manifest_hash"),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(row, indent=2))
    with (output_dir / "comparison.csv").open("w", newline="") as h:
        writer = csv.DictWriter(h, row.keys())
        writer.writeheader()
        writer.writerow(row)
    (output_dir / "comparison.md").write_text(
        "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in row.items())
    )
    return row
