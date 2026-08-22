"""Flower server startup isolated from experiment orchestration."""

import json
from pathlib import Path
from typing import Any

from .orchestration import configured_rounds
from .strategy import create_fedavg


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def start_server(
    address: str, config: dict[str, Any], strategy: Any = None, run_dir: Path | None = None
) -> None:
    import flwr as fl

    if strategy is None and run_dir is not None:
        from fl_brats.models.factory import create_model

        from .orchestration import _load_initial_weights
        from .strategy import create_research_fedavg

        run_dir.mkdir(parents=True, exist_ok=True)
        for relative in ("checkpoints", "metrics", "communication", "logs"):
            (run_dir / relative).mkdir(exist_ok=True)
        (run_dir / "resolved_config.yaml").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8"
        )
        model = create_model(config.get("model", {}))
        _load_initial_weights(model, config)
        strategy = create_research_fedavg(config, run_dir=run_dir, model=model)

    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(
            num_rounds=configured_rounds(config),
            round_timeout=config.get("round_timeout_seconds"),
        ),
        strategy=strategy or create_fedavg(config),
    )
    if run_dir is not None and strategy is not None and hasattr(strategy, "state"):
        from fl_brats.metrics.reporting import (
            communication_summary,
            convergence_rounds,
            dice_communication_curve,
            pareto_frontier,
            per_client_summary,
        )

        evaluations = _jsonl(run_dir / "metrics" / "evaluation_rounds.jsonl")
        communication_records = _jsonl(run_dir / "communication" / "rounds.jsonl")
        client_records = _jsonl(run_dir / "metrics" / "clients.jsonl")
        evaluated = [
            item
            for item in client_records
            if isinstance(item.get("mean_dice"), (int, float))
        ]
        final_round = max((int(item["round_number"]) for item in evaluated), default=None)
        final_clients = [
            item
            for item in evaluated
            if final_round is not None and int(item["round_number"]) == final_round
        ]
        modality_rows = _jsonl(run_dir / "metrics" / "modality_combinations.jsonl")
        final_modality_round = max((int(item["round"]) for item in modality_rows), default=None)
        final_modality_rows = [
            item for item in modality_rows
            if final_modality_round is not None and int(item["round"]) == final_modality_round
        ]
        per_client_combinations: dict[str, dict[str, dict[str, float]]] = {}
        for item in final_modality_rows:
            combinations: dict[str, dict[str, float]] = {}
            for key, value in item["metrics"].items():
                _, combination, metric = key.split("__", 2)
                combinations.setdefault(combination, {})[metric] = float(value)
            per_client_combinations[str(item["client_id"])] = combinations
        aggregate_combinations: dict[str, dict[str, float]] = {}
        combination_names = sorted(
            {name for combinations in per_client_combinations.values() for name in combinations}
        )
        for name in combination_names:
            metric_names = sorted(
                {
                    key
                    for combinations in per_client_combinations.values()
                    for key in combinations[name]
                }
            )
            aggregate_combinations[name] = {
                key: sum(
                    combinations[name][key] for combinations in per_client_combinations.values()
                )
                / len(per_client_combinations)
                for key in metric_names
            }
        communication = communication_summary(communication_records)
        curve = dice_communication_curve(evaluations, communication_records)
        summary = {
            "schema_version": 1,
            "status": strategy.state.status if strategy.state.status != "running" else "completed",
            "completed_rounds": strategy.state.completed_round,
            "best_round": strategy.state.best_round,
            "best_validation_dice": strategy.state.best_metric,
            "final_validation_dice": (
                evaluations[-1].get("validation_mean_dice") if evaluations else None
            ),
            "metrics": {
                "per_client": per_client_summary(final_clients),
                "missing_modality_robustness": {
                    "all_four_modalities": aggregate_combinations.get("t1+t1ce+t2+flair"),
                    "combinations": aggregate_combinations,
                    "per_client_combinations": per_client_combinations,
                }
                if aggregate_combinations
                else None,
            },
            "convergence": convergence_rounds(evaluations),
            "efficiency": communication
            | {
                "dice_vs_cumulative_communication": curve,
                "pareto_frontier": pareto_frontier(curve),
            },
        }
        (run_dir / "communication" / "summary.json").write_text(
            json.dumps(summary["efficiency"], indent=2), encoding="utf-8"
        )
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
