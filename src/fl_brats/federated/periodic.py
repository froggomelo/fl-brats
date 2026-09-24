"""Post-training periodic evaluation of saved round checkpoints.

Full metrics (HD95, sensitivity) and the 15-combination robustness pass are
reporting-only: model selection and early stopping use natural Dice every
round. Computing them during training blocked the whole federation for hours
every tenth round, so training records natural Dice only and this job
evaluates every ``every``-th round checkpoint afterwards with the same
evaluator. The checkpoints are the same models, so the metrics are the same.

A run that was resumed spreads its checkpoints over several run directories,
so the whole chain is accepted; output goes to the newest one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from fl_brats.data.hashing import sha256_file, sha256_payload
from fl_brats.models.factory import create_model

from .evaluation import HeldOutEvaluator
from .research import atomic_write_json
from .revalidation import atomic_jsonl, discover_checkpoints, evaluation_lock, run_values


def lineage_checkpoints(run_dirs: list[Path]) -> dict[int, Path]:
    """Round checkpoints across a resume chain; a later directory wins a round."""
    rounds: dict[int, Path] = {}
    for run_dir in run_dirs:
        try:
            found = discover_checkpoints(run_dir)
        except ValueError:
            continue
        rounds.update(dict(found))
    if not rounds:
        raise ValueError(f"no round checkpoints in {', '.join(map(str, run_dirs))}")
    return rounds


def evaluate_periodic(
    run_dirs: list[Path],
    split_path: Path | None = None,
    *,
    every: int = 10,
    index_path: Path | None = None,
    device: str | None = None,
    hd95_workers: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if every < 1:
        raise ValueError("every must be a positive number of rounds")
    run_dirs = [path.resolve() for path in run_dirs]
    output = run_dirs[-1] / "periodic_evaluation"
    checkpoints = lineage_checkpoints(run_dirs)
    selected = sorted(number for number in checkpoints if number % every == 0)
    values = run_values(run_dirs[-1], device)
    if hd95_workers is not None:
        values.setdefault("evaluation", {})["hd95_workers"] = hd95_workers
    evaluator = HeldOutEvaluator(values, split_path, index_path=index_path)
    robustness = evaluator.robustness_enabled
    metadata = evaluator.metadata | {
        "run_dirs": [str(path) for path in run_dirs],
        "every_rounds": every,
        "robustness": robustness,
        "metrics": ["dice", "hd95", "sensitivity"],
    }
    # Execution details must not change the identity of a resumable evaluation.
    identity_fields = {key: value for key, value in metadata.items() if key != "cadence"}
    identity = sha256_payload(identity_fields)
    plan = {
        "output": str(output),
        "rounds": selected,
        "robustness": robustness,
        "validation_subject_count": len(evaluator.subjects),
        "evaluation_identity": identity,
        "dry_run": dry_run,
    }
    if dry_run:
        return plan
    evaluator.prepare()
    with evaluation_lock(output):
        metadata_path = output / "metadata.json"
        if metadata_path.exists():
            if json.loads(metadata_path.read_text())["evaluation_identity"] != identity:
                raise ValueError("periodic evaluation identity changed; use a separate output")
        else:
            atomic_write_json(metadata_path, metadata | {"evaluation_identity": identity})
        rounds_path = output / "rounds.jsonl"
        combinations_path = output / "modality_combinations.jsonl"
        rows = [json.loads(line) for line in rounds_path.read_text().splitlines()] if rounds_path.exists() else []
        done = {row["round"] for row in rows}
        # A round's combination rows are written before its round row, so rows
        # for a round without a round row are from an interrupted evaluation.
        combination_rows = (
            [
                row
                for row in map(json.loads, combinations_path.read_text().splitlines())
                if row["round"] in done
            ]
            if combinations_path.exists()
            else []
        )
        for row in rows:
            if sha256_file(checkpoints[row["round"]]) != row["checkpoint_hash"]:
                raise ValueError(f"checkpoint changed since evaluation: round {row['round']}")
        atomic_write_json(output / "state.json", {"status": "running", "completed": len(rows)})
        for number in selected:
            if number in done:
                continue
            checkpoint = checkpoints[number]
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if int(payload["round"]) != number:
                raise ValueError(f"checkpoint round disagrees with filename: {checkpoint}")
            model = create_model(payload["model_config"])
            model.load_state_dict(payload["model_state"], strict=True)
            metrics, combinations = evaluator.run(model, expensive=True, robustness=robustness)
            if not math.isfinite(metrics["mean_dice"]):
                raise ValueError(f"non-finite validation score for {checkpoint}")
            combination_rows += [
                {"round": number, "combination": key, **value} for key, value in combinations.items()
            ]
            atomic_jsonl(combinations_path, combination_rows)
            rows.append(
                {"round": number, **metrics, "checkpoint": str(checkpoint), "checkpoint_hash": sha256_file(checkpoint)}
            )
            rows.sort(key=lambda row: row["round"])
            atomic_jsonl(rounds_path, rows)
            atomic_write_json(output / "state.json", {"status": "running", "completed": len(rows)})
            print(
                f"Periodic round={number} mean_dice={metrics['mean_dice']:.6f} "
                f"mean_hd95={metrics.get('mean_hd95', float('nan')):.3f} "
                f"seconds={metrics['evaluation_seconds']:.0f}",
                flush=True,
            )
            del model
        summary = plan | {"rounds_evaluated": len(rows)}
        atomic_write_json(output / "summary.json", summary)
        atomic_write_json(output / "state.json", {"status": "completed", "completed": len(rows)})
        return summary
