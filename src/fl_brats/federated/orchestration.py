"""Real Flower orchestration shared by the CLI local/server/client commands."""
# ruff: noqa: E501, E701, E702

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import flwr as fl
import torch

from fl_brats.data.brats import build_dataloader, build_dataset, records_from_client_manifest
from fl_brats.data.hashing import sha256_file
from fl_brats.data.transforms import build_transforms
from fl_brats.losses.segmentation import create_loss
from fl_brats.metrics.reporting import (
    communication_summary,
    convergence_rounds,
    dice_communication_curve,
    pareto_frontier,
    per_client_summary,
)
from fl_brats.models.factory import create_model, model_report

from .model_client import ModelFedAvgClient
from .parameters import state_to_arrays
from .strategy import create_research_fedavg


def _append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _mean_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(
        {key for row in rows for key, value in row.items() if isinstance(value, (int, float))}
    )
    return {
        key: sum(float(row[key]) for row in rows if isinstance(row.get(key), (int, float)))
        / sum(isinstance(row.get(key), (int, float)) for row in rows)
        for key in keys
    }


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _load_initial_weights(model: torch.nn.Module, values: dict[str, Any]) -> str | None:
    path_value = values.get("initial_weights") or values.get("initial_checkpoint")
    if not path_value:
        return None
    path = Path(path_value)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"initial weights do not contain a state dict: {path}")
    model.load_state_dict(state, strict=True)
    return sha256_file(path)


def configured_rounds(values: dict[str, Any]) -> int:
    training = values.get("training", {})
    if isinstance(training, dict) and training.get("max_rounds") is not None:
        return int(training["max_rounds"])
    return int(values.get("max_rounds", values.get("num_rounds", 50)))


def build_client(values: dict[str, Any], client_id: str, partition_root: Path) -> ModelFedAvgClient:
    """Create a client using only its own manifest file."""
    path = partition_root / f"{client_id}.json"
    if not path.exists():
        raise ValueError(f"unknown client ID {client_id!r}: missing {path}")
    records = records_from_client_manifest(path)
    if not records:
        raise ValueError(f"client {client_id!r} has an empty partition")
    data = values.get("data", {})
    model_config = values.get("model", {})
    pin_memory = values.get("pin_memory")
    train = build_dataloader(
        build_dataset(records, build_transforms(data, True)),
        int(values.get("local_batch_size", values.get("batch_size", 1))),
        True,
        int(values.get("num_workers", 0)),
        int(values.get("seed", 42)),
        bool(pin_memory) if pin_memory is not None else None,
    )
    valid = build_dataloader(
        build_dataset(records, build_transforms(data, False)),
        int(values.get("local_batch_size", values.get("batch_size", 1))),
        False,
        int(values.get("num_workers", 0)),
        int(values.get("seed", 42)),
        bool(pin_memory) if pin_memory is not None else None,
    )
    model = create_model(model_config)
    _load_initial_weights(model, values)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(values.get("learning_rate", 1e-4)),
        weight_decay=float(values.get("weight_decay", 0.0)),
    )
    missing = values.get("missing_modalities", {})
    missing_evaluation = missing.get("evaluation", {}) if isinstance(missing, dict) else {}
    return ModelFedAvgClient(
        model,
        train_loader=train,
        validation_loader=valid,
        loss_fn=create_loss(values.get("loss", {})),
        optimizer=optimizer,
        device=_device(str(values.get("device", "auto"))),
        local_epochs=int(values.get("local_epochs", 1)),
        client_id=client_id,
        strategy=str(values.get("strategy", "fedavg")),
        fedprox_mu=float(values.get("fedprox", {}).get("mu", values.get("fedprox_mu", 0.0))),
        quantization_config=values.get("quantization", {}),
        evaluate_modality_robustness=bool(missing.get("enabled", False))
        and bool(missing_evaluation.get("evaluate_all_non_empty_combinations", True)),
    )


def run_local_federation(values: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run an actual gRPC Flower server and local client processes (threads)."""
    partition_root = Path(
        values.get("partition_root") or values.get("data", {}).get("partition_root", "")
    )
    if not partition_root.exists():
        raise ValueError(f"partition_root does not exist: {partition_root}")
    client_ids = [f"client_{i}" for i in range(int(values.get("num_clients", 0)))]
    if not client_ids:
        client_ids = [path.stem for path in sorted(partition_root.glob("client_*.json"))]
    clients = [build_client(values, client_id, partition_root) for client_id in client_ids]
    if len(clients) < int(values.get("min_available_clients", len(clients))):
        raise ValueError("fewer usable client partitions than min_available_clients")
    run_dir.mkdir(parents=True, exist_ok=False)
    for relative in ("checkpoints", "metrics", "communication", "logs"):
        (run_dir / relative).mkdir()
    (run_dir / "resolved_config.yaml").write_text(
        json.dumps(values, indent=2, default=str), encoding="utf-8"
    )
    port = int(values.get("server_port", 0))
    if port == 0:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
    address = str(values.get("server_address") or f"127.0.0.1:{port}")
    server_model = create_model(values.get("model", {}))
    initial_weight_hash = _load_initial_weights(server_model, values)
    strategy = create_research_fedavg(values, run_dir=run_dir, model=server_model)
    history: list[Any] = []

    def server() -> None:
        history.append(
            fl.server.start_server(
                server_address=address,
                config=fl.server.ServerConfig(num_rounds=configured_rounds(values)),
                strategy=strategy,
            )
        )

    started = time.monotonic()
    server_thread = threading.Thread(target=server, daemon=True)
    server_thread.start()
    time.sleep(0.2)  # bounded startup grace; gRPC retries connection establishment.
    errors: list[str] = []

    def connect(client: ModelFedAvgClient) -> None:
        try:
            fl.client.start_numpy_client(server_address=address, client=client)
        except Exception as error:
            errors.append(f"{client.client_id}: {error}")

    threads = [threading.Thread(target=connect, args=(client,), daemon=True) for client in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    server_thread.join(timeout=30)
    if server_thread.is_alive() or errors:
        raise RuntimeError(
            "Flower local run failed: " + "; ".join(errors or ["server did not finish"])
        )
    if strategy.state.latest_checkpoint is None:
        raise RuntimeError("Flower server completed without a successful aggregated round")
    final = strategy.checkpoints.save_alias(Path(strategy.state.latest_checkpoint), "final.pt")
    strategy.state.status = "completed"
    strategy.writer.write_state(strategy.state)
    model = server_model
    global_parameters = state_to_arrays(model)[1]
    evaluations = [client.evaluate(global_parameters, {})[2] for client in clients]
    mean = sum(item["mean_dice"] for item in evaluations) / len(evaluations)
    rounds = configured_rounds(values)
    evaluation_rounds = _read_jsonl(run_dir / "metrics" / "evaluation_rounds.jsonl")
    communication_rounds = _read_jsonl(run_dir / "communication" / "rounds.jsonl")
    communication = communication_summary(communication_rounds)
    curve = dice_communication_curve(evaluation_rounds, communication_rounds)
    fairness = per_client_summary(evaluations)
    modality_results: dict[str, Any] | None = None
    missing = values.get("missing_modalities", {})
    evaluation_config = missing.get("evaluation", {}) if isinstance(missing, dict) else {}
    if bool(missing.get("enabled", False)) and bool(
        evaluation_config.get("evaluate_all_non_empty_combinations", True)
    ):
        per_client_combinations: dict[str, dict[str, dict[str, float]]] = {}
        for client, metrics in zip(clients, evaluations, strict=True):
            combinations: dict[str, dict[str, float]] = {}
            for key, value in metrics.items():
                if key.startswith("modality__") and isinstance(value, (int, float)):
                    _, combination, metric = key.split("__", 2)
                    combinations.setdefault(combination, {})[metric] = float(value)
            per_client_combinations[client.client_id] = combinations
        combination_keys = sorted(next(iter(per_client_combinations.values())).keys())
        aggregate = {
            key: _mean_metric_rows([result[key] for result in per_client_combinations.values()])
            for key in combination_keys
        }
        modality_results = {
            "natural_missing_distribution": fairness,
            "all_four_modalities": aggregate.get("t1+t1ce+t2+flair"),
            "combinations": aggregate,
            "per_client_combinations": per_client_combinations,
        }
        (run_dir / "metrics" / "modality_robustness.json").write_text(
            json.dumps(modality_results, indent=2), encoding="utf-8"
        )
    summary = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": strategy.state.status if strategy.state.status != "running" else "completed",
        "model_name": values.get("model", {}).get("name"),
        "model_family": values.get("model", {}).get("name"),
        "seed": values.get("seed"),
        "strategy": values.get("strategy", "fedavg"),
        "local_epochs": values.get("local_epochs"),
        "batch_size": values.get("local_batch_size", values.get("batch_size")),
        "max_rounds": rounds,
        "actual_rounds": strategy.state.completed_round,
        "optimizer": values.get("optimizer", {}).get("name", "adam"),
        "learning_rate": values.get("learning_rate"),
        "fedprox_mu": values.get("fedprox", {}).get("mu", values.get("fedprox_mu"))
        if values.get("strategy") == "fedprox"
        else None,
        "initial_weight_hash": initial_weight_hash,
        "number_of_clients": len(clients),
        "completed_rounds": strategy.state.completed_round,
        "best_validation_dice": strategy.state.best_metric,
        "best_round": strategy.state.best_round,
        "final_validation_dice": mean,
        "metrics": {
            "macro_dice": mean,
            "segmentation": _mean_metric_rows(
                [
                    {
                        key: value
                        for key, value in item.items()
                        if not key.startswith("modality__")
                    }
                    for item in evaluations
                ]
            ),
            "per_client": fairness,
            "missing_modality_robustness": modality_results,
        },
        "convergence": convergence_rounds(evaluation_rounds),
        "efficiency": {
            "total_wall_clock_time": time.monotonic() - started,
            **communication,
            "dice_vs_cumulative_communication": curve,
            "pareto_frontier": pareto_frontier(curve),
        },
        "artifacts": {
            "final_checkpoint": str(final),
            "best_checkpoint": strategy.state.best_checkpoint,
        },
        "model": model_report(model, values.get("model", {})),
    }
    (run_dir / "communication" / "summary.json").write_text(
        json.dumps(summary["efficiency"], indent=2), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
