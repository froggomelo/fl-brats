"""Server-side experiment provenance for classic Flower FedAvg."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from flwr.common import FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from .downlink import DownlinkEncoder
from .parameters import load_arrays, state_to_arrays
from .quantization import aggregate_deltas, dequantized_delta, deserialize_client_update
from .sparsification import (
    aggregate_sparse_updates,
    is_sparse_parameter_payload,
    sparse_update_from_ndarrays,
)
from fl_brats.models.state import checkpoint_state

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
    client_update_l2: float | None
    normalized_client_update_l2: float | None
    status: str
    error: str | None = None
    wt_hd95: float | None = None
    tc_hd95: float | None = None
    et_hd95: float | None = None
    wt_sensitivity: float | None = None
    tc_sensitivity: float | None = None
    et_sensitivity: float | None = None
    # Parts of train_loss (fl_brats.training.engine.LossComponents); the
    # FedAMM terms stay null for the other methods.
    seg_loss: float | None = None
    distill_loss: float | None = None
    distill_loss_t1: float | None = None
    distill_loss_t1ce: float | None = None
    distill_loss_t2: float | None = None
    distill_loss_flair: float | None = None
    align_loss: float | None = None
    aux_loss_weighted: float | None = None
    aux_fraction: float | None = None


LOSS_COMPONENTS = (
    "seg_loss",
    "distill_loss",
    "distill_loss_t1",
    "distill_loss_t1ce",
    "distill_loss_t2",
    "distill_loss_flair",
    "align_loss",
    "aux_loss_weighted",
    "aux_fraction",
)


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
        strategy_state: Mapping[str, Any] | None = None,
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
            "model_state": checkpoint_state(self.model),
            "experiment_state": asdict(state),
            "metrics": dict(metrics),
            "strategy_state": dict(strategy_state or {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(f"refusing to overwrite round checkpoint: {path}")
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def save_alias(self, source: Path, name: str) -> Path:
        destination = self.run_dir / "checkpoints" / name
        temporary = destination.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
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
        self,
        *,
        writer: RoundArtifactWriter,
        checkpoints: CheckpointManager,
        early_stopping_patience: int | None = None,
        minimum_rounds: int = 0,
        initial_state: ExperimentState | None = None,
        round_offset: int = 0,
        client_diagnostics: bool = False,
        downlink_bit_width: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if early_stopping_patience is not None and int(early_stopping_patience) < 1:
            raise ValueError("early_stopping_patience must be a positive integer")
        self.client_diagnostics = client_diagnostics
        self._evaluated_rounds: set[int] = set()
        self._round_timing: dict[int, dict[str, Any]] = {}
        self.writer = writer
        self.checkpoints = checkpoints
        self.state = initial_state or ExperimentState()
        self.round_offset = max(int(round_offset), 0)
        self.early_stopping_patience = (
            int(early_stopping_patience) if early_stopping_patience is not None else None
        )
        self.minimum_rounds = max(int(minimum_rounds), 0)
        self._round_started: dict[int, float] = {}
        self._current_arrays = state_to_arrays(checkpoints.model)[1]
        self._keys = state_to_arrays(checkpoints.model)[0]
        # Quantized broadcast (federated/downlink.py); None sends the FP32 model.
        self.downlink = (
            DownlinkEncoder(self._keys, downlink_bit_width) if downlink_bit_width is not None else None
        )

    def checkpoint_strategy_state(self) -> dict[str, Any]:
        """Server-side state beyond the model that a resumed run must restore."""
        return {}

    def restore_strategy_state(self, state: Mapping[str, Any]) -> None:
        del state

    def global_round(self, server_round: int) -> int:
        """Translate Flower's per-process round into the experiment-wide round."""
        return self.round_offset + int(server_round)

    def _should_stop_early(self) -> bool:
        return (
            self.early_stopping_patience is not None
            and self.state.completed_round >= self.minimum_rounds
            and self.state.best_round is not None
            and self.state.completed_round - self.state.best_round >= self.early_stopping_patience
        )

    def _mark_early_stopped(self) -> None:
        if self.state.status != "early_stopped":
            self.state.status = "early_stopped"
            self.writer.write_state(self.state)

    def configure_fit(self, server_round: int, parameters: Any, client_manager: Any) -> Any:
        if self._should_stop_early():
            self._mark_early_stopped()
            return []
        self._round_started[server_round] = time.perf_counter()
        instructions = super().configure_fit(server_round, parameters, client_manager)
        if self.downlink is None or not instructions:
            return instructions
        # One payload per round, the same for every client (a broadcast).
        broadcast = ndarrays_to_parameters(
            self.downlink.encode(parameters_to_ndarrays(parameters), self.global_round(server_round))
        )
        return [(proxy, FitIns(broadcast, fit_ins.config)) for proxy, fit_ins in instructions]

    def configure_evaluate(self, server_round: int, parameters: Any, client_manager: Any) -> Any:
        if not self.client_diagnostics:
            return []
        if self._should_stop_early():
            self._mark_early_stopped()
            return []
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(
        self, server_round: int, results: Any, failures: Any
    ) -> tuple[Parameters | None, dict[str, Any]]:
        aggregation_started = time.perf_counter()
        global_round = self.global_round(server_round)
        records = self._records(global_round, results, failures)
        if self._uses_quantized_payloads(results):
            parameters, metrics = self._aggregate_quantized_fit(global_round, results)
        elif self._uses_sparse_payloads(results):
            parameters, metrics = self._aggregate_sparse_fit(results)
        else:
            parameters, metrics = super().aggregate_fit(server_round, results, failures)
            parameters = self.rebase_dense_aggregate(parameters)
        if parameters is None:
            self.writer.write_client_records(records)
            return None, metrics
        return self._persist_aggregate(
            server_round, parameters, metrics, records, aggregation_started
        )

    def rebase_dense_aggregate(self, parameters: Parameters | None) -> Parameters | None:
        """Move a dense average of client models from the broadcast view to the FP32 model.

        Clients trained from the quantized view, so their mean is view + mean update;
        the server applies the mean update to its FP32 model instead. Compressed
        uplinks already carry updates and are applied to the FP32 model directly.
        """
        if parameters is None or self.downlink is None or self.downlink.view is None:
            return parameters
        averaged = parameters_to_ndarrays(parameters)
        return ndarrays_to_parameters(
            [
                (current + (mean - view)).astype(current.dtype)
                for current, mean, view in zip(
                    self._current_arrays, averaged, self.downlink.view, strict=True
                )
            ]
        )

    def _persist_aggregate(self, server_round, parameters, metrics, records, aggregation_started):
        """Shared by FedAvg, compressed transports, and modality-weighted FedAMM."""
        global_round = self.global_round(server_round)
        self._current_arrays = parameters_to_ndarrays(parameters)
        aggregation_seconds = time.perf_counter() - aggregation_started
        record = {
            "schema_version": SCHEMA_VERSION,
            "round": global_round,
            "fit_aggregation_serialization_seconds": aggregation_seconds,
            "max_client_training_seconds": self._max(records, "fit_seconds"),
            "successful_clients": sum(r.status == "completed" for r in records),
            "mean_client_drift": self._mean(records, "client_update_l2"),
            "max_client_drift": self._max(records, "client_update_l2"),
            "mean_normalized_client_drift": self._mean(records, "normalized_client_update_l2"),
            **{f"mean_{name}": self._mean(records, name) for name in LOSS_COMPONENTS},
        }
        next_state = ExperimentState(**asdict(self.state))
        next_state.completed_round = global_round
        next_state.uploaded_bytes += sum(r.upload_payload_bytes for r in records)
        next_state.downloaded_bytes += sum(r.download_payload_bytes for r in records)
        started = time.perf_counter()
        checkpoint = self.checkpoints.save_round(
            global_round, parameters, next_state, record, self.checkpoint_strategy_state()
        )
        record["checkpoint_writing_seconds"] = time.perf_counter() - started
        next_state.latest_checkpoint = str(checkpoint)
        self._round_timing[server_round] = record
        self.writer.write_client_records(records)
        self.writer.write_communication_record(
            {
                "round": global_round,
                "uploaded_bytes": sum(r.upload_payload_bytes for r in records),
                "downloaded_bytes": sum(r.download_payload_bytes for r in records),
                "network_transmission_seconds": None,
                "communication_and_coordination_seconds": None,
                "measurement_note": (
                    "No inferred network timing; measured components in metrics/rounds.jsonl."
                ),
            }
        )
        self.state = next_state
        self.writer.write_state(self.state)
        return parameters, metrics

    def evaluate(self, server_round: int, parameters: Parameters):
        """Flower invokes this after aggregate_fit and before distributed diagnostics.

        Skip Flower's initial round-zero evaluation and rounds without aggregation.
        Only this held-out natural result can select a checkpoint or stop training.
        """
        global_round = self.global_round(server_round)
        if (
            server_round == 0
            or server_round in self._evaluated_rounds
            or server_round not in self._round_timing
        ):
            return None
        self._evaluated_rounds.add(server_round)
        result = (
            self.evaluate_fn(global_round, parameters_to_ndarrays(parameters), {})
            if self.evaluate_fn
            else None
        )
        timing = self._round_timing.pop(server_round)
        if result is not None:
            loss, metrics = result
            score = float(metrics["mean_dice"])
            if not math.isfinite(score):
                raise ValueError("non-finite held-out validation mean Dice")
            row = {
                "round": global_round,
                "loss": loss,
                **metrics,
                "checkpoint": self.state.latest_checkpoint,
                "selection_criterion": "held_out_natural_validation_mean_dice",
            }
            self.writer._append(self.writer.run_dir / "metrics/validation_rounds.jsonl", row)
            for key in (
                "natural_validation_seconds",
                "expensive_metrics_seconds",
                "robustness_seconds",
            ):
                timing[key] = metrics.get(key)
            if self.state.best_metric is None or score > self.state.best_metric:
                started = time.perf_counter()
                self.state.best_checkpoint = str(
                    self.checkpoints.save_alias(Path(self.state.latest_checkpoint), "best.pt")
                )
                timing["checkpoint_writing_seconds"] += time.perf_counter() - started
                self.state.best_metric, self.state.best_round = score, global_round
                print(
                    f"Selected best.pt round={global_round} held-out mean_dice={score:.6f}",
                    flush=True,
                )
            self.writer.write_state(self.state)
        timing["round_wall_seconds"] = time.perf_counter() - self._round_started.pop(
            server_round, time.perf_counter()
        )
        self.writer.write_round_record(timing)
        return result

    def _uses_sparse_payloads(self, results: Any) -> bool:
        if not results:
            return False
        decoded = [
            is_sparse_parameter_payload(parameters_to_ndarrays(result.parameters))
            for _, result in results
        ]
        if any(decoded) and not all(decoded):
            raise ValueError("cannot mix sparse and dense client update payloads in one round")
        return all(decoded)

    @staticmethod
    def _uses_quantized_payloads(results: Any) -> bool:
        if not results:
            return False
        arrays = [parameters_to_ndarrays(result.parameters) for _, result in results]
        return all(
            len(items) == 1 and items[0].dtype == np.uint8 and items[0].ndim == 1
            for items in arrays
        )

    def _aggregate_quantized_fit(
        self, server_round: int, results: Any
    ) -> tuple[Parameters, dict[str, Any]]:
        base = {
            key: torch.from_numpy(value.copy())
            for key, value in zip(self._keys, self._current_arrays, strict=True)
        }
        updates = []
        achieved = []
        for _, result in results:
            arrays = parameters_to_ndarrays(result.parameters)
            update = deserialize_client_update(arrays[0].tobytes())
            if update.round_index != server_round:
                raise ValueError("quantized update round does not match server round")
            updates.append((dequantized_delta(update), int(result.num_examples)))
            if update.achieved_average_bit_width is not None:
                achieved.append(update.achieved_average_bit_width)
        aggregated = aggregate_deltas(base, updates)
        arrays = [aggregated[key].detach().cpu().numpy() for key in self._keys]
        return ndarrays_to_parameters(arrays), {
            "mean_achieved_bit_width": sum(achieved) / len(achieved) if achieved else 0.0
        }

    def _aggregate_sparse_fit(self, results: Any) -> tuple[Parameters | None, dict[str, Any]]:
        updates = []
        for _, result in results:
            update = sparse_update_from_ndarrays(parameters_to_ndarrays(result.parameters))
            updates.append((update, int(result.num_examples)))
        arrays = aggregate_sparse_updates(updates, self._keys, self._current_arrays)
        return ndarrays_to_parameters(arrays), {}

    def aggregate_evaluate(
        self, server_round: int, results: Any, failures: Any
    ) -> tuple[float | None, dict[str, Any]]:
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        global_round = self.global_round(server_round)
        records = self._evaluation_records(global_round, results, failures)
        mean_dice = self._weighted_mean(records, "mean_dice")
        record = {
            "schema_version": SCHEMA_VERSION,
            "round": global_round,
            "validation_loss": loss,
            "validation_wt_dice": self._weighted_mean(records, "dice_wt"),
            "validation_tc_dice": self._weighted_mean(records, "dice_tc"),
            "validation_et_dice": self._weighted_mean(records, "dice_et"),
            "validation_mean_dice": mean_dice,
            "validation_wt_hd95": self._weighted_mean(records, "wt_hd95"),
            "validation_tc_hd95": self._weighted_mean(records, "tc_hd95"),
            "validation_et_hd95": self._weighted_mean(records, "et_hd95"),
            "validation_wt_sensitivity": self._weighted_mean(records, "wt_sensitivity"),
            "validation_tc_sensitivity": self._weighted_mean(records, "tc_sensitivity"),
            "validation_et_sensitivity": self._weighted_mean(records, "et_sensitivity"),
            "successful_clients": sum(item.status == "completed" for item in records),
            "failed_clients": sum(item.status == "failed" for item in records),
        }
        if failures:
            record["failures"] = [str(failure) for failure in failures]
        record = {
            key.replace("validation_", "training_partition_diagnostic_"): value
            for key, value in record.items()
        }
        self.writer._append(self.writer.run_dir / "metrics/client_diagnostics.jsonl", record)
        for proxy, result in results:
            modality_metrics = {
                key: value for key, value in result.metrics.items() if key.startswith("modality__")
            }
            if modality_metrics:
                self.writer._append(
                    self.writer.run_dir / "metrics" / "modality_combinations.jsonl",
                    {
                        "round": global_round,
                        "client_id": str(result.metrics.get("client_id", proxy.cid)),
                        "metrics": modality_metrics,
                    },
                )
        return loss, metrics

    @staticmethod
    def _mean(records: list[ClientRoundRecord], field: str) -> float | None:
        values = [
            getattr(record, field) for record in records if getattr(record, field) is not None
        ]
        return sum(values) / len(values) if values else None

    @staticmethod
    def _max(records: list[ClientRoundRecord], field: str) -> float | None:
        values = [
            getattr(record, field) for record in records if getattr(record, field) is not None
        ]
        return max(values) if values else None

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
                replace(
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
                        self._float(metric, "client_update_l2"),
                        self._float(metric, "normalized_client_update_l2"),
                        "completed",
                    ),
                    wt_hd95=self._float(metric, "wt_hd95"),
                    tc_hd95=self._float(metric, "tc_hd95"),
                    et_hd95=self._float(metric, "et_hd95"),
                    wt_sensitivity=self._float(metric, "wt_sensitivity"),
                    tc_sensitivity=self._float(metric, "tc_sensitivity"),
                    et_sensitivity=self._float(metric, "et_sensitivity"),
                    **{name: self._float(metric, name) for name in LOSS_COMPONENTS},
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
                    None,
                    None,
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
                replace(
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
                        None,
                        None,
                        "completed",
                    ),
                    wt_hd95=self._float(metric, "wt_hd95"),
                    tc_hd95=self._float(metric, "tc_hd95"),
                    et_hd95=self._float(metric, "et_hd95"),
                    wt_sensitivity=self._float(metric, "wt_sensitivity"),
                    tc_sensitivity=self._float(metric, "tc_sensitivity"),
                    et_sensitivity=self._float(metric, "et_sensitivity"),
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
                    None,
                    None,
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
