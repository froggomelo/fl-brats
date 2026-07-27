"""Validated U-Net-reference comparisons for SAM architectural replacement."""

import json
from pathlib import Path
from typing import Any


CONTROL_KEYS = ("dataset_index_hash", "split_manifest_hash", "partition_manifest_hash", "number_of_clients", "communication_rounds", "local_epochs", "evaluation_policy")


def validate_comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    mismatches = [key for key in CONTROL_KEYS if baseline.get(key) != candidate.get(key)]
    if candidate.get("prompt_mode") == "oracle" or candidate.get("oracle_prompt"):
        mismatches.append("oracle prompts are excluded")
    return mismatches


def comparison_row(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    mismatches = validate_comparison(baseline, candidate)
    if "oracle prompts are excluded" in mismatches:
        raise ValueError("oracle-prompt runs cannot be included in primary comparisons")
    row: dict[str, Any] = {"reference_model": "federated_unet_baseline", "candidate_model": candidate.get("condition"), "control_mismatches": validate_comparison(baseline, candidate)}
    for region in ("wt", "tc", "et", "mean"):
        key = f"test_{region}_dice"
        row[f"{region}_dice"] = candidate.get(key)
        row[f"{region}_dice_difference_from_unet"] = (candidate.get(key) or 0) - (baseline.get(key) or 0)
    base_bytes, candidate_bytes = baseline.get("cumulative_communication_bytes", 0), candidate.get("cumulative_communication_bytes", 0)
    row["communication_change_bytes"] = candidate_bytes - base_bytes
    row["mean_dice_relative_to_unet"] = (row["mean_dice"] or 0) / (baseline.get("test_mean_dice") or 1)
    return row


def compare_sam_to_baseline(baseline_path: Path, sam1_path: Path | None, sam2_path: Path | None, output_dir: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text())
    candidates = [("federated_sam1", sam1_path), ("federated_sam2", sam2_path)]
    rows = []
    for condition, path in candidates:
        if path:
            candidate = json.loads(path.read_text()); candidate.setdefault("condition", condition)
            rows.append(comparison_row(baseline, candidate))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "architecture_replacement.json").write_text(json.dumps(rows, indent=2))
    return {"comparisons": rows}
