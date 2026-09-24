"""Real Flower orchestration shared by the CLI local/server/client commands."""
# ruff: noqa: E501, E701, E702

from __future__ import annotations

import json
import shutil
import socket
import threading
import time
from pathlib import Path
from typing import Any

import flwr as fl
import torch

from fl_brats.data.brats import build_dataloader, build_dataset, records_from_client_manifest
from fl_brats.data.cohorts import read_split, split_path_for, validate_client_manifest
from fl_brats.data.hashing import sha256_file
from fl_brats.data.transforms import build_transforms
from fl_brats.losses.segmentation import create_loss
from fl_brats.models.factory import create_model

from .evaluation import HeldOutEvaluator, evaluation_policy
from .finalization import finalize_federation
from .model_client import ModelFedAvgClient
from .research import ExperimentState
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
    path_value = (
        values.get("resume")
        or values.get("resume_checkpoint")
        or values.get("initial_weights")
        or values.get("initial_checkpoint")
    )
    if not path_value:
        return None
    path = Path(path_value)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"initial weights do not contain a state dict: {path}")
    model.load_state_dict(state, strict=True)
    return sha256_file(path)


def _copy_jsonl_through_round(source: Path, destination: Path, completed_round: int) -> None:
    if not source.exists():
        return
    rows = _read_jsonl(source)
    retained = [
        row
        for row in rows
        if int(row.get("round", row.get("round_number", 0))) <= completed_round
    ]
    if not retained:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained),
        encoding="utf-8",
    )


def load_resume_state(
    model: torch.nn.Module, values: dict[str, Any], run_dir: Path
) -> ExperimentState | None:
    """Restore a checkpoint and seed a new continuation directory with its history."""
    path_value = values.get("resume") or values.get("resume_checkpoint")
    if not path_value:
        return None
    checkpoint = Path(path_value).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        raise ValueError(f"resume checkpoint is not a federated round checkpoint: {checkpoint}")
    checkpoint_round = int(payload.get("round", 0))
    if checkpoint_round < 1:
        raise ValueError(f"resume checkpoint has an invalid round: {checkpoint}")
    model.load_state_dict(payload["model_state"], strict=True)
    state_data = payload.get("experiment_state")
    if not isinstance(state_data, dict):
        raise ValueError(f"resume checkpoint lacks experiment_state: {checkpoint}")

    source_run = checkpoint.parent.parent
    if source_run.resolve() == run_dir.resolve():
        raise ValueError("resume requires a new run directory")
    state_path = source_run / "state.json"
    if state_path.exists():
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        if int(persisted.get("completed_round", -1)) == checkpoint_round:
            state_data = persisted
    state = ExperimentState(
        **{key: value for key, value in state_data.items() if key in ExperimentState.__dataclass_fields__}
    )
    state.completed_round = checkpoint_round
    state.status = "running"
    state.latest_checkpoint = str(checkpoint)

    for relative in (
        "metrics/clients.jsonl",
        "metrics/rounds.jsonl",
        "metrics/evaluation_rounds.jsonl",
        "metrics/validation_rounds.jsonl",
        "metrics/modality_combinations.jsonl",
        "communication/rounds.jsonl",
    ):
        _copy_jsonl_through_round(source_run / relative, run_dir / relative, checkpoint_round)

    metadata_path = source_run / "evaluation_metadata.json"
    if not metadata_path.exists():
        import warnings
        warnings.warn("Historical training-partition selection state reset for held-out validation.", stacklevel=2)
        state.best_round = state.best_metric = state.best_checkpoint = None
        return state
    prior_metadata = json.loads(metadata_path.read_text())
    current_metadata = HeldOutEvaluator(values).metadata
    for key in ("validation_subject_list_hash", "test_subject_list_hash",
                "preprocessing_config_hash", "modality_policy", "natural_masks", "metric_policy"):
        if prior_metadata.get(key) != current_metadata.get(key):
            raise ValueError(f"cannot resume with a different evaluation {key}")
    # Checkpoint payloads precede evaluation. Recover selection from the committed
    # held-out rows, never a best.pt alias that may refer to a later source round.
    rows = _read_jsonl(run_dir / "metrics/validation_rounds.jsonl")
    state.best_round = state.best_metric = state.best_checkpoint = None
    if rows:
        best = max(rows, key=lambda row: row["mean_dice"])
        source_best = source_run / "checkpoints" / f"round_{int(best['round']):04d}.pt"
        destination = run_dir / "checkpoints/best.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_best, destination)
        state.best_round, state.best_metric = int(best["round"]), float(best["mean_dice"])
        state.best_checkpoint = str(destination)
    return state


def configured_rounds(values: dict[str, Any]) -> int:
    training = values.get("training", {})
    if isinstance(training, dict) and training.get("max_rounds") is not None:
        return int(training["max_rounds"])
    return int(values.get("max_rounds", values.get("num_rounds", 50)))


def rounds_to_run(values: dict[str, Any], state: ExperimentState | None = None) -> int:
    completed = state.completed_round if state is not None else 0
    remaining = configured_rounds(values) - completed
    if remaining < 1:
        raise ValueError(
            f"experiment already completed {completed} of {configured_rounds(values)} rounds"
        )
    return remaining


def build_client(values: dict[str, Any], client_id: str, partition_root: Path) -> ModelFedAvgClient:
    """Create a client using only its own manifest file."""
    path = partition_root / f"{client_id}.json"
    if not path.exists():
        raise ValueError(f"unknown client ID {client_id!r}: missing {path}")
    split = read_split(split_path_for(values))
    ids = validate_client_manifest(path, split)
    print(f"{client_id}: training only, {len(ids)} subjects; disjoint from validation/test", flush=True)
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
    valid = None
    if evaluation_policy(values)["client_diagnostics"]:
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
    values = dict(values)
    values["initial_weight_hash"] = _load_initial_weights(server_model, values)
    resume_state = load_resume_state(server_model, values, run_dir)
    evaluator = HeldOutEvaluator(values)
    evaluator.prepare()
    strategy = create_research_fedavg(
        values,
        run_dir=run_dir,
        model=server_model,
        evaluate_fn=evaluator.callback(server_model, run_dir),
        initial_state=resume_state,
        round_offset=resume_state.completed_round if resume_state else 0,
    )
    started = time.monotonic()
    errors: list[str] = []

    def connect(client: ModelFedAvgClient) -> None:
        try:
            fl.client.start_numpy_client(server_address=address, client=client)
        except Exception as error:
            errors.append(f"{client.client_id}: {error}")

    threads = [threading.Thread(target=connect, args=(client,), daemon=True) for client in clients]
    for thread in threads:
        thread.start()
    # Flower installs process signal handlers: its server must run on the main thread.
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=rounds_to_run(values, resume_state)),
        strategy=strategy,
    )
    for thread in threads:
        thread.join(timeout=30)
    if errors or any(thread.is_alive() for thread in threads):
        raise RuntimeError("Flower local run failed: " + "; ".join(errors or ["client did not finish"]))
    if strategy.state.latest_checkpoint is None:
        raise RuntimeError("Flower server completed without a successful aggregated round")
    return finalize_federation(strategy, evaluator, values, run_dir, time.monotonic() - started)
