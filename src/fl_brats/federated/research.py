"""Server-side experiment provenance for classic Flower FedAvg."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from flwr.common import Parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from .parameters import load_arrays, state_to_arrays

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ClientRoundRecord:
    round_number: int
    client_id: str
    num_examples: int
    train_loss: float | None
    val_loss: float | None
    dice_wt: float | None
    dice_tc: float | None
    dice_et: float | None
    mean_dice: float | None
    fit_seconds: float | None
    upload_payload_bytes: int
    download_payload_bytes: int
    status: str
    error: str | None = None


@dataclass
class ExperimentState:
    schema_version: int = SCHEMA_VERSION
    completed_round: int = 0
    best_round: int | None = None
    best_metric: float | None = None
    uploaded_bytes: int = 0
    downloaded_bytes: int = 0
    latest_checkpoint: str | None = None
    best_checkpoint: str | None = None
    status: str = "running"


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace JSON metadata in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class RoundArtifactWriter:
    """The server-only append-only artifact sink."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def _append(self, path: Path, record: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_client_records(self, records: list[ClientRoundRecord]) -> None:
        for record in records:
            self._append(self.run_dir / "metrics" / "clients.jsonl", asdict(record))

    def write_round_record(self, record: Mapping[str, Any]) -> None:
        self._append(self.run_dir / "metrics" / "rounds.jsonl", record)

    def write_evaluation_record(self, record: Mapping[str, Any]) -> None:
        self._append(self.run_dir / "metrics" / "evaluation_rounds.jsonl", record)

    def write_communication_record(self, record: Mapping[str, Any]) -> None:
        self._append(self.run_dir / "communication" / "rounds.jsonl", record)

    def write_state(self, state: ExperimentState) -> None:
        atomic_write_json(self.run_dir / "state.json", asdict(state))

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        atomic_write_json(self.run_dir / "summary.json", summary)


class CheckpointManager:
    """Convert named Flower arrays into portable model checkpoints."""

    def __init__(self, run_dir: Path, model: torch.nn.Module, model_config: dict[str, Any]) -> None:
        self.run_dir, self.model, self.model_config = run_dir, model, model_config

    def save_round(
        self,
        round_number: int,
        parameters: Parameters,
        state: ExperimentState,
        metrics: Mapping[str, Any],
    ) -> Path:
        arrays = parameters_to_ndarrays(parameters)
        keys, _ = state_to_arrays(self.model)
        self._validate(arrays)
        load_arrays(self.model, keys, arrays)
        path = self.run_dir / "checkpoints" / f"round_{round_number:04d}.pt"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "round": round_number,
            "model_config": self.model_config,
            "model_state": self.model.state_dict(),
            "experiment_state": asdict(state),
            "metrics": dict(metrics),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def save_alias(self, source: Path, name: str) -> Path:
        destination = self.run_dir / "checkpoints" / name
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(source.read_bytes())
        temporary.replace(destination)
        return destination

    def _validate(self, arrays: list[np.ndarray]) -> None:
        expected = state_to_arrays(self.model)[1]
        if len(arrays) != len(expected):
            raise ValueError("aggregated parameter tensor count differs from server model")
        for index, (array, reference) in enumerate(zip(arrays, expected, strict=True)):
            if array.shape != reference.shape or not np.isfinite(array).all():
                raise ValueError(f"invalid aggregated parameter tensor at index {index}")


class ResearchFedAvgStrategy(FedAvg):
    """FedAvg with a small successful-round persistence transaction."""

    def __init__(
        self, *, writer: RoundArtifactWriter, checkpoints: CheckpointManager, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.writer, self.checkpoints, self.state = writer, checkpoints, ExperimentState()
        self._round_started: dict[int, float] = {}

    def aggregate_fit(
        self, server_round: int, results: Any, failures: Any
    ) -> tuple[Parameters | None, dict[str, Any]]:
        self._round_started[server_round] = time.perf_counter()
        records = self._records(server_round, results, failures)
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is None:
            self.writer.write_client_records(records)
            return None, metrics
        elapsed = time.perf_counter() - self._round_started.pop(server_round)
        round_record = {
            "schema_version": SCHEMA_VERSION,
            "round": server_round,
            "round_wall_seconds": elapsed,
            "successful_clients": sum(record.status == "completed" for record in records),
        }
        next_state = ExperimentState(**asdict(self.state))
        next_state.completed_round = server_round
        next_state.uploaded_bytes += sum(record.upload_payload_bytes for record in records)
        next_state.downloaded_bytes += sum(record.download_payload_bytes for record in records)
        checkpoint = self.checkpoints.save_round(server_round, parameters, next_state, round_record)
        next_state.latest_checkpoint = str(checkpoint)
        self.writer.write_client_records(records)
        self.writer.write_round_record(round_record)
        self.writer.write_communication_record(
            {
                "round": server_round,
                "uploaded_bytes": sum(r.upload_payload_bytes for r in records),
                "downloaded_bytes": sum(r.download_payload_bytes for r in records),
            }
        )
        self.state = next_state
        self.writer.write_state(self.state)
        return parameters, metrics

    def aggregate_evaluate(
        self, server_round: int, results: Any, failures: Any
    ) -> tuple[float | None, dict[str, Any]]:
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        records = self._evaluation_records(server_round, results, failures)
        mean_dice = self._weighted_mean(records, "mean_dice")
        record = {
            "schema_version": SCHEMA_VERSION,
            "round": server_round,
            "validation_loss": loss,
            "validation_wt_dice": self._weighted_mean(records, "dice_wt"),
            "validation_tc_dice": self._weighted_mean(records, "dice_tc"),
            "validation_et_dice": self._weighted_mean(records, "dice_et"),
            "validation_mean_dice": mean_dice,
            "successful_clients": sum(item.status == "completed" for item in records),
            "failed_clients": sum(item.status == "failed" for item in records),
        }
        if failures:
            record["failures"] = [str(failure) for failure in failures]
        self.writer.write_evaluation_record(record)
        next_state = ExperimentState(**asdict(self.state))
        if mean_dice is not None and next_state.latest_checkpoint is not None and (
            next_state.best_metric is None or mean_dice > next_state.best_metric
        ):
            next_state.best_metric, next_state.best_round = mean_dice, server_round
            next_state.best_checkpoint = str(
                self.checkpoints.save_alias(Path(next_state.latest_checkpoint), "best.pt")
            )
            self.state = next_state
            self.writer.write_state(self.state)
        return loss, metrics

    @staticmethod
    def _mean(records: list[ClientRoundRecord], field: str) -> float | None:
        values = [
            getattr(record, field) for record in records if getattr(record, field) is not None
        ]
        return sum(values) / len(values) if values else None

    @staticmethod
    def _weighted_mean(records: list[ClientRoundRecord], field: str) -> float | None:
        weighted = [
            (record.num_examples, getattr(record, field))
            for record in records
            if getattr(record, field) is not None and record.num_examples > 0
        ]
        total = sum(num_examples for num_examples, _ in weighted)
        if total <= 0:
            return None
        return sum(num_examples * float(value) for num_examples, value in weighted) / total

    def _records(self, round_number: int, results: Any, failures: Any) -> list[ClientRoundRecord]:
        records: list[ClientRoundRecord] = []
        for proxy, result in results:
            metric = result.metrics
            records.append(
                ClientRoundRecord(
                    round_number,
                    str(metric.get("client_id", proxy.cid)),
                    result.num_examples,
                    self._float(metric, "train_loss"),
                    self._float(metric, "loss"),
                    self._float(metric, "wt_dice"),
                    self._float(metric, "tc_dice"),
                    self._float(metric, "et_dice"),
                    self._float(metric, "mean_dice"),
                    self._float(metric, "local_training_seconds"),
                    int(metric.get("upload_numpy_array_bytes", 0)),
                    int(metric.get("download_payload_bytes", 0)),
                    "completed",
                )
            )
        for failure in failures:
            records.append(
                ClientRoundRecord(
                    round_number,
                    "unknown",
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    "failed",
                    str(failure),
                )
            )
        return records

    def _evaluation_records(
        self, round_number: int, results: Any, failures: Any
    ) -> list[ClientRoundRecord]:
        records: list[ClientRoundRecord] = []
        for proxy, result in results:
            metric = result.metrics
            records.append(
                ClientRoundRecord(
                    round_number,
                    str(metric.get("client_id", proxy.cid)),
                    result.num_examples,
                    None,
                    self._float(metric, "loss"),
                    self._float(metric, "wt_dice"),
                    self._float(metric, "tc_dice"),
                    self._float(metric, "et_dice"),
                    self._float(metric, "mean_dice"),
                    None,
                    0,
                    0,
                    "completed",
                )
            )
        for failure in failures:
            records.append(
                ClientRoundRecord(
                    round_number,
                    "unknown",
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    "failed",
                    str(failure),
                )
            )
        return records

    @staticmethod
    def _float(metrics: Mapping[str, Any], key: str) -> float | None:
        value = metrics.get(key)
        return float(value) if isinstance(value, (float, int)) else None


def weighted_average_metrics(results: list[tuple[int, Mapping[str, Any]]]) -> dict[str, float]:
    """Aggregate numeric Flower client metrics by each client's example count."""
    total = sum(num_examples for num_examples, _ in results)
    if total <= 0:
        return {}
    keys = sorted(
        {
            key
            for _, metrics in results
            for key, value in metrics.items()
            if isinstance(value, (float, int))
        }
    )
    return {
        key: sum(float(metrics.get(key, 0.0)) * num_examples for num_examples, metrics in results)
        / total
        for key in keys
    }
