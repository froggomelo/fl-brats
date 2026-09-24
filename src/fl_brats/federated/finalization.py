"""Shared successful completion for local and distributed Flower runs."""

import json
import time
from pathlib import Path
from typing import Any

from fl_brats.metrics.reporting import (
    communication_summary,
    convergence_rounds,
    dice_communication_curve,
    pareto_frontier,
)

from .research import atomic_write_json


def finalize_federation(
    strategy: Any, evaluator: Any, values: dict[str, Any], run_dir: Path, training_seconds: float
) -> dict[str, Any]:
    if strategy.state.latest_checkpoint is None:
        raise RuntimeError("Flower completed without a successful aggregated round")
    final = strategy.checkpoints.save_alias(Path(strategy.state.latest_checkpoint), "final.pt")
    started = time.perf_counter()
    for name, checkpoint in (("best", strategy.state.best_checkpoint), ("final", str(final))):
        if checkpoint and evaluator.policy[f"evaluate_{name}_checkpoint"]:
            evaluator.evaluate_checkpoint(
                Path(checkpoint), run_dir / "validation_evaluation" / name
            )
    strategy.state.status = (
        "completed" if strategy.state.status != "early_stopped" else "early_stopped"
    )
    strategy.writer.write_state(strategy.state)
    path = run_dir / "metrics/validation_rounds.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    communication_path = run_dir / "communication/rounds.jsonl"
    communication = (
        communication_summary(
            [json.loads(line) for line in communication_path.read_text().splitlines()]
        )
        if communication_path.exists()
        else {}
    )
    legacy_rows = [{**row, "validation_mean_dice": row["mean_dice"]} for row in rows]
    communication_rows = (
        [json.loads(line) for line in communication_path.read_text().splitlines()]
        if communication_path.exists()
        else []
    )
    curve = dice_communication_curve(legacy_rows, communication_rows)
    communication["measurement_note"] = (
        "Payload bytes measured directly; network/coordination time unavailable. "
        "Measured round components are in metrics/rounds.jsonl."
    )
    communication["total_communication_and_coordination_seconds"] = None
    final_row = rows[-1] if rows and rows[-1]["round"] == strategy.state.completed_round else {}
    summary = {
        "schema_version": 2,
        "initial_weight_hash": values.get("initial_weight_hash"),
        "number_of_clients": values.get("num_clients"),
        "local_epochs": values.get("local_epochs"),
        "batch_size": values.get("local_batch_size", values.get("batch_size")),
        "learning_rate": values.get("learning_rate"),
        "convergence": convergence_rounds(legacy_rows),
        "run_id": run_dir.name,
        "status": strategy.state.status,
        "strategy": values.get("strategy", "fedavg"),
        "seed": values.get("seed"),
        "model_name": values.get("model", {}).get("name"),
        "model_family": values.get("model", {}).get("name"),
        "metrics": {"macro_dice": final_row.get("mean_dice"), "segmentation": final_row},
        "completed_rounds": strategy.state.completed_round,
        "actual_rounds": strategy.state.completed_round,
        "best_round": strategy.state.best_round,
        "best_validation_dice": strategy.state.best_metric,
        "final_validation_dice": rows[-1]["mean_dice"]
        if rows and rows[-1]["round"] == strategy.state.completed_round
        else None,
        "selection_criterion": "held_out_natural_validation_mean_dice",
        "evaluation_metadata": evaluator.metadata,
        "efficiency": {
            **communication,
            "dice_vs_cumulative_communication": curve,
            "pareto_frontier": pareto_frontier(curve),
            "training_wall_seconds": training_seconds,
            "post_training_evaluation_seconds": time.perf_counter() - started,
        },
        "artifacts": {
            "best_checkpoint": strategy.state.best_checkpoint,
            "final_checkpoint": str(final),
        },
    }
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "communication/summary.json", summary["efficiency"])
    return summary
