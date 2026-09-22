"""FedAMM Flower strategy with modality-weighted encoder aggregation."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays

from fl_brats.data.modalities import MODALITIES

from .parameters import state_keys
from .research import ResearchFedAvgStrategy


def _weighted_average(items: list[tuple[np.ndarray, float]]) -> np.ndarray:
    total = sum(weight for _, weight in items)
    if total <= 0:
        total = float(len(items))
        return sum(array for array, _ in items) / total
    return sum(array * weight for array, weight in items) / total


def aggregate_fedamm_arrays(keys: list[str], results: Any) -> list[np.ndarray]:
    """Aggregate modality encoders by modality counts and shared tensors by examples."""
    output: list[np.ndarray] = []
    client_arrays = [(parameters_to_ndarrays(fit_res.parameters), fit_res) for _, fit_res in results]
    for index, key in enumerate(keys):
        modality = next((m for m in MODALITIES if key.startswith(f"modality_encoders.{m}.")), None)
        weighted = []
        for arrays, fit_res in client_arrays:
            if modality is None:
                weight = float(fit_res.num_examples)
            else:
                weight = float(fit_res.metrics.get(f"fedamm_modality_count_{modality}", 0.0))
            weighted.append((arrays[index], weight))
        output.append(_weighted_average(weighted))
    return output


def aggregate_sample_count_arrays(results: Any) -> list[np.ndarray]:
    """Ordinary sample-count-weighted FedAvg over every tensor."""
    output: list[np.ndarray] = []
    client_arrays = [(parameters_to_ndarrays(fit_res.parameters), fit_res) for _, fit_res in results]
    for index in range(len(client_arrays[0][0])):
        output.append(
            _weighted_average([(arrays[index], float(fit_res.num_examples)) for arrays, fit_res in client_arrays])
        )
    return output


def uses_modality_weighted_aggregation(model: Any) -> bool:
    return bool(getattr(model, "modality_weighted_aggregation", True))


class GlobalPrototypeLibrary:
    """Server-side prototype store grouped by modality-combination ID."""

    def __init__(self, clusters_per_combination: int = 1) -> None:
        self.clusters_per_combination = max(int(clusters_per_combination), 1)
        self.centres: dict[int, list[list[list[float]]]] = {}

    def update_from_results(self, results: Any) -> dict[str, Any]:
        grouped: dict[int, list[np.ndarray]] = {}
        for _, fit_res in results:
            payload = fit_res.metrics.get("fedamm_prototypes_json")
            if not isinstance(payload, str):
                continue
            for key, item in json.loads(payload).items():
                grouped.setdefault(int(key), []).append(np.asarray(item["prototype"], dtype=np.float32))
        for combo_id, values in grouped.items():
            self.centres[combo_id] = [centre.tolist() for centre in kmeans(np.stack(values), self.clusters_per_combination)]
        return {
            "prototype_combinations": len(self.centres),
            "prototype_cluster_count": sum(len(v) for v in self.centres.values()),
        }

    def dumps(self) -> str:
        return json.dumps({str(key): value for key, value in self.centres.items()}, sort_keys=True)


def kmeans(values: np.ndarray, k: int, iterations: int = 8) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    k = min(k, flat.shape[0])
    centres = flat[:k].copy()
    for _ in range(iterations):
        distances = ((flat[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for index in range(k):
            assigned = flat[labels == index]
            if len(assigned):
                centres[index] = assigned.mean(axis=0)
    return centres.reshape((k, *values.shape[1:]))


class FedAMMStrategy(ResearchFedAvgStrategy):
    """Research strategy implementing FedAMM aggregation and prototype clustering."""

    def __init__(self, *, model: Any, clusters_per_combination: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.keys = state_keys(model)
        self.modality_weighted_aggregation = uses_modality_weighted_aggregation(model)
        self.prototype_library = GlobalPrototypeLibrary(clusters_per_combination)

    def aggregate_fit(self, server_round: int, results: Any, failures: Any) -> tuple[Parameters | None, dict[str, Any]]:
        if failures and not self.accept_failures:
            self.writer.write_client_records(self._records(server_round, results, failures))
            return None, {}
        if self._uses_sparse_payloads(results) or self._uses_quantized_payloads(results):
            # Compressed transports are decoded by the shared research strategy.
            # Prototype state remains active; sparse layer-wise modality weighting
            # is not representable after global top-k selection.
            self.prototype_library.update_from_results(results)
            return super().aggregate_fit(server_round, results, failures)
        global_round = self.global_round(server_round)
        records = self._records(global_round, results, failures)
        if not results:
            self.writer.write_client_records(records)
            return None, {}
        if self.modality_weighted_aggregation:
            parameters = ndarrays_to_parameters(aggregate_fedamm_arrays(self.keys, results))
        else:
            parameters = ndarrays_to_parameters(aggregate_sample_count_arrays(results))
        prototype_metrics = self.prototype_library.update_from_results(results)
        import time
        from dataclasses import asdict
        from .research import ExperimentState

        round_metric = self._mean(records, "dice_wt")
        elapsed = time.perf_counter() - self._round_started.pop(
            server_round, time.perf_counter()
        )
        max_client_training = self._max(records, "fit_seconds")
        coordination = (
            max(elapsed - max_client_training, 0.0)
            if max_client_training is not None
            else None
        )
        round_record = {
            "schema_version": 1,
            "round": global_round,
            "client_wt_dice_mean": round_metric,
            "round_wall_seconds": elapsed,
            "max_client_training_seconds": max_client_training,
            "communication_and_coordination_seconds": coordination,
            "successful_clients": sum(record.status == "completed" for record in records),
            **prototype_metrics,
        }
        next_state = ExperimentState(**asdict(self.state))
        next_state.completed_round = global_round
        next_state.uploaded_bytes += sum(record.upload_payload_bytes for record in records)
        next_state.downloaded_bytes += sum(record.download_payload_bytes for record in records)
        checkpoint = self.checkpoints.save_round(global_round, parameters, next_state, round_record)
        next_state.latest_checkpoint = str(checkpoint)
        if round_metric is not None and (
            next_state.best_metric is None or round_metric > next_state.best_metric
        ):
            next_state.best_metric, next_state.best_round = round_metric, global_round
            next_state.best_checkpoint = str(self.checkpoints.save_alias(checkpoint, "best.pt"))
        self.writer.write_client_records(records)
        self.writer.write_round_record(round_record)
        self.writer.write_communication_record(
            {
                "round": global_round,
                "uploaded_bytes": sum(r.upload_payload_bytes for r in records),
                "downloaded_bytes": sum(r.download_payload_bytes for r in records),
                "network_transmission_seconds": None,
                "communication_and_coordination_seconds": coordination,
                "measurement_note": (
                    "Round wall time minus the slowest reported client training time; "
                    "includes serialization, RPC, scheduling, waiting, and aggregation overhead."
                ),
            }
        )
        self.state = next_state
        self.writer.write_state(self.state)
        return parameters, prototype_metrics

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: Any) -> Any:
        instructions = super().configure_fit(server_round, parameters, client_manager)
        payload = self.prototype_library.dumps()
        if payload == "{}":
            return instructions
        for _, fit_ins in instructions:
            fit_ins.config["fedamm_global_prototypes_json"] = payload
        return instructions
