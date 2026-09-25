"""Segmentation-aware Flower client used by local and distributed runs."""
# ruff: noqa: E501

from __future__ import annotations

import math
import time
from collections.abc import Sized
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from fl_brats.data.modalities import MODALITIES, modality_counts
from fl_brats.training.engine import evaluate, train_local_epochs
from fl_brats.training.evaluation import evaluate_modality_combinations

from .client import FedAvgClient
from .communication import payload_metrics, quantized_payload_metrics
from .parameters import load_arrays, state_keys, state_to_arrays
from .quantization import (
    BitWidthAllocator,
    LayerQuantizationStats,
    eligible_state,
    quantize_update,
    serialize_client_update,
)
from .sparsification import LayerQuantizedValues, create_sparse_update, sparse_update_to_ndarrays


def modality_count_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    masks = [item["modality_mask"] for item in records if "modality_mask" in item]
    if not masks:
        return {}
    counts = modality_counts(torch.stack([mask.bool() for mask in masks]))
    return {f"fedamm_modality_count_{modality}": float(counts[modality]) for modality in MODALITIES}


class ModelFedAvgClient(FedAvgClient):
    """A deterministic, model-owning Flower client for segmentation workloads.

    ``train_fn``/``evaluate_fn`` retain the small callback API used by older
    callers.  Supplying loaders and an optimizer instead delegates to the
    shared training engine, which is the normal production path.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        keys: list[str] | None = None,
        train_fn: Callable[[], dict[str, Any]] | None = None,
        evaluate_fn: Callable[[], dict[str, float]] | None = None,
        examples: int | None = None,
        *,
        train_loader: DataLoader[Any] | None = None,
        validation_loader: DataLoader[Any] | None = None,
        loss_fn: Any = None,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | str = "cpu",
        local_epochs: int = 1,
        client_id: str = "unknown",
        strategy: str = "fedavg",
        fedprox_mu: float = 0.0,
        quantization_config: dict[str, Any] | None = None,
        evaluate_modality_robustness: bool = False,
        state_dir: Path | None = None,
        resume_state_path: Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        # Optimizer moments and compression residuals live in this process between
        # rounds. Persisting them per round lets a resumed run continue exactly
        # where the interrupted one stopped instead of silently resetting them.
        self.state_dir = state_dir
        self._resume_state_path = resume_state_path
        self.model = model.to(self.device)
        self.keys = keys or state_keys(model)
        self.train_loader, self.validation_loader = train_loader, validation_loader
        self.loss_fn, self.optimizer = loss_fn, optimizer
        if strategy not in {"fedavg", "fedprox", "fedamm"}:
            raise ValueError(f"unsupported federated strategy: {strategy}")
        self.local_epochs, self.client_id = local_epochs, client_id
        self.strategy, self.fedprox_mu = strategy, fedprox_mu
        self.quantization_config = quantization_config or {}
        self.evaluate_modality_robustness = evaluate_modality_robustness
        self._evaluation_config: dict[str, Any] = {}
        self._sparsification_residual: torch.Tensor | None = None
        self._quantization_residuals: dict[str, torch.Tensor] = {}
        self._global_arrays_before_fit: list[np.ndarray] | None = None
        self._proximal_global_parameters: list[torch.Tensor] | None = None
        if examples is None:
            examples = len(cast(Sized, train_loader.dataset)) if train_loader is not None else 0
        if examples <= 0:
            raise ValueError(f"client {client_id!r} has an empty training partition")
        if train_fn is None:
            if train_loader is None or loss_fn is None or optimizer is None:
                raise ValueError("train_fn or train_loader, loss_fn, and optimizer are required")
            train_fn = self._train
        if evaluate_fn is None:
            evaluate_fn = self._evaluate if validation_loader is not None else self._evaluation_disabled
        super().__init__(self.model, self.keys, train_fn, evaluate_fn, examples)

    def client_state_path(self, directory: Path, server_round: int) -> Path:
        return directory / f"{self.client_id}_round_{server_round:04d}.pt"

    def _save_client_state(self, server_round: int) -> None:
        if self.state_dir is None or server_round < 1:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.client_state_path(self.state_dir, server_round)
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "round": server_round,
                "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
                "sparsification_residual": self._sparsification_residual,
                "quantization_residuals": self._quantization_residuals,
            },
            temporary,
        )
        temporary.replace(path)
        for stale in self.state_dir.glob(f"{self.client_id}_round_*.pt"):
            if int(stale.stem.rsplit("_", 1)[1]) <= server_round - 3:
                stale.unlink(missing_ok=True)

    def _restore_client_state(self) -> None:
        path, self._resume_state_path = self._resume_state_path, None
        if path is None:
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if self.optimizer is not None and payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        self._sparsification_residual = payload.get("sparsification_residual")
        self._quantization_residuals = dict(payload.get("quantization_residuals") or {})
        print(f"{self.client_id}: restored client state from {path}", flush=True)

    def _evaluation_disabled(self) -> dict[str, float]:
        raise RuntimeError("client evaluation disabled; use server held-out validation")

    def get_model_parameters(self) -> list[np.ndarray]:
        return state_to_arrays(self.model)[1]

    def validate_parameter_payload(self, parameters: list[np.ndarray]) -> None:
        # load_arrays performs exact name/count/shape and dtype-preserving checks.
        if not all(isinstance(item, np.ndarray) for item in parameters):
            raise ValueError("Flower parameter payload must contain NumPy arrays")
        load_arrays(self.model, self.keys, parameters)

    def set_model_parameters(self, parameters: list[np.ndarray]) -> None:
        self.validate_parameter_payload(parameters)

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return self.get_model_parameters()

    def _train(self) -> dict[str, Any]:
        assert (
            self.train_loader is not None
            and self.loss_fn is not None
            and self.optimizer is not None
        )
        started = time.monotonic()
        result = train_local_epochs(
            self.model,
            self.train_loader,
            self.loss_fn,
            self.optimizer,
            self.device,
            epochs=self.local_epochs,
            proximal_mu=self.fedprox_mu if self.strategy == "fedprox" else 0.0,
            proximal_global_parameters=self._proximal_global_parameters,
        )
        result["local_training_seconds"] = time.monotonic() - started
        if self.train_loader is not None:
            result.update(modality_count_metrics(getattr(self.train_loader.dataset, "data", [])))
        return result

    def _evaluate(self) -> dict[str, float]:
        assert self.validation_loader is not None and self.loss_fn is not None
        expensive = bool(self._evaluation_config.get("compute_expensive_metrics", False))
        metric_names = ("dice", "hd95", "sensitivity") if expensive else ("dice",)
        result = evaluate(
            self.model,
            self.validation_loader,
            self.loss_fn,
            self.device,
            metric_names=metric_names,
        )
        run_robustness = bool(
            self._evaluation_config.get("evaluate_modality_robustness", False)
        )
        if self.evaluate_modality_robustness and run_robustness:
            combinations = evaluate_modality_combinations(
                self.model,
                self.validation_loader,
                self.loss_fn,
                self.device,
                metric_names=("dice", "hd95", "sensitivity"),
            )
            for combination, metrics in combinations.items():
                for key, value in metrics.items():
                    result[f"modality__{combination}__{key}"] = value
        return result

    @staticmethod
    def _finite(metrics: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in metrics.items():
            output[key] = value if isinstance(value, str) else float(value)
        numeric = [value for value in output.values() if isinstance(value, float)]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("client produced non-finite loss or metric")
        return output

    def _sparse_bit_width_allocation(self) -> Any:
        """Mixed-precision allocation over the layers of a sparse update.

        Same FedMPQ-inspired allocator and client budget as dense mixed mode,
        with each layer weighted by what it transmits: its selected entries and
        their L2 norm, so the average bit width is over transmitted values.
        """
        mixed = self.quantization_config.get("mixed", {})
        patterns = [str(item).lower() for item in mixed.get("protected_layer_patterns", [])]
        budget = float(
            mixed.get("client_average_bit_width_budgets", {}).get(
                self.client_id, mixed.get("default_average_bit_width_budget", 4.0)
            )
        )

        def allocate(statistics: dict[str, tuple[int, float]]) -> dict[str, int]:
            return BitWidthAllocator().allocate(
                layer_statistics={
                    name: LayerQuantizationStats(count, norm + 1e-12)
                    for name, (count, norm) in statistics.items()
                },
                candidate_bit_widths=list(mixed.get("candidate_bit_widths", [2, 4, 8])),
                average_budget=budget,
                protected_layers={
                    name for name in statistics if any(pattern in name.lower() for pattern in patterns)
                },
                protected_minimum=int(mixed.get("protected_layer_minimum_bit_width", 8)),
            ).assignments

        return allocate

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        self._restore_client_state()
        self.set_model_parameters(parameters)
        self._global_arrays_before_fit = [item.copy() for item in parameters]
        self._proximal_global_parameters = (
            [
                parameter.detach().clone().to(self.device)
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            if self.strategy == "fedprox"
            else None
        )
        if "fedamm_global_prototypes_json" in config:
            self.model.global_prototypes_json = config["fedamm_global_prototypes_json"]
        metrics = self._finite(self.train_fn())
        arrays = self.get_model_parameters()
        metrics["client_id"] = self.client_id  # Flower scalar contract permits strings.
        metrics["download_payload_bytes"] = float(sum(item.nbytes for item in parameters))
        upload_arrays = arrays
        sparse = self.quantization_config.get("sparsification", {})
        sparse_enabled = bool(sparse.get("enabled", False))
        if sparse_enabled:
            if self._global_arrays_before_fit is None:
                raise ValueError("sparse update requires cached global parameters")
            quant_enabled = bool(self.quantization_config.get("enabled", False))
            quant_mode = str(self.quantization_config.get("mode", "none"))
            bit_width = int(self.quantization_config.get("fixed", {}).get("bit_width", 8))
            if quant_enabled and quant_mode not in ("fixed", "mixed"):
                raise ValueError(f"unsupported quantization mode with sparsification: {quant_mode}")
            update, self._sparsification_residual = create_sparse_update(
                client_id=self.client_id,
                round_index=int(config.get("server_round", 0)),
                keys=self.keys,
                local_arrays=arrays,
                global_arrays=self._global_arrays_before_fit,
                sample_count=self.examples,
                fraction=float(sparse.get("fraction", 1.0)),
                residual=self._sparsification_residual
                if bool(sparse.get("error_feedback", True))
                else None,
                quantize_values=quant_enabled,
                bit_width=bit_width,
                scheme=self.quantization_config.get("scheme", "symmetric"),
                granularity=self.quantization_config.get("granularity", "per_tensor"),
                rounding=self.quantization_config.get("rounding", "nearest"),
                seed=(
                    int(config["server_round"])
                    if self.quantization_config.get("rounding") == "stochastic"
                    else None
                ),
                allocate_bit_widths=(
                    self._sparse_bit_width_allocation()
                    if quant_enabled and quant_mode == "mixed"
                    else None
                ),
            )
            upload_arrays = sparse_update_to_ndarrays(update)
            metrics.update(
                {
                    "upload_scalar_parameter_count": float(update.total_parameters),
                    "upload_transmitted_parameter_count": float(update.selected_count),
                    "upload_sparsification_fraction": float(update.fraction),
                    "upload_actual_sparse_payload_bytes": float(update.payload_bytes),
                    "upload_metadata_bytes": float(update.metadata_bytes),
                    "upload_numpy_array_bytes": float(sum(item.nbytes for item in upload_arrays)),
                    "upload_compression_ratio": float(
                        (update.total_parameters * 4)
                        / max(sum(item.nbytes for item in upload_arrays), 1)
                    ),
                }
            )
            if isinstance(update.values, LayerQuantizedValues):
                metrics["upload_achieved_average_bit_width"] = float(update.values.average_bit_width)
        elif bool(self.quantization_config.get("enabled", False)):
            if self._global_arrays_before_fit is None:
                raise ValueError("quantized delta requires cached global parameters")
            local_state = {
                key: torch.from_numpy(value.copy())
                for key, value in zip(self.keys, arrays, strict=True)
            }
            global_state = {
                key: torch.from_numpy(value.copy())
                for key, value in zip(self.keys, self._global_arrays_before_fit, strict=True)
            }
            eligible = eligible_state(local_state)
            mode = str(self.quantization_config.get("mode", "fixed"))
            if mode == "fixed":
                width = int(self.quantization_config.get("fixed", {}).get("bit_width", 8))
                assignment = {key: width for key in eligible}
            elif mode == "mixed":
                mixed = self.quantization_config.get("mixed", {})
                patterns = [str(item).lower() for item in mixed.get("protected_layer_patterns", [])]
                protected = {
                    key for key in eligible if any(pattern in key.lower() for pattern in patterns)
                }
                budget = float(
                    mixed.get("client_average_bit_width_budgets", {}).get(
                        self.client_id, mixed.get("default_average_bit_width_budget", 4.0)
                    )
                )
                allocation = BitWidthAllocator().allocate(
                    layer_statistics={
                        key: LayerQuantizationStats(
                            value.numel(), float(value.float().norm()) + 1e-12
                        )
                        for key, value in eligible.items()
                    },
                    candidate_bit_widths=list(mixed.get("candidate_bit_widths", [2, 4, 8])),
                    average_budget=budget,
                    protected_layers=protected,
                    protected_minimum=int(mixed.get("protected_layer_minimum_bit_width", 8)),
                )
                assignment = allocation.assignments
            else:
                raise ValueError(f"unsupported enabled quantization mode: {mode}")
            update, residuals = quantize_update(
                client_id=self.client_id,
                round_index=int(config.get("server_round", 0)),
                local_state=local_state,
                global_state=global_state,
                sample_count=self.examples,
                bit_width_assignment=assignment,
                scheme=self.quantization_config.get("scheme", "symmetric"),
                granularity=self.quantization_config.get("granularity", "per_tensor"),
                rounding=self.quantization_config.get("rounding", "nearest"),
                seed=int(config.get("server_round", 0)),
                residuals=self._quantization_residuals
                if bool(self.quantization_config.get("error_feedback", {}).get("enabled", False))
                else None,
            )
            self._quantization_residuals = residuals
            wire = serialize_client_update(update)
            upload_arrays = [np.frombuffer(wire, dtype=np.uint8).copy()]
            metrics.update(
                {
                    f"upload_{key}": float(value)
                    for key, value in quantized_payload_metrics(update).items()
                    if isinstance(value, (int, float))
                }
            )
            metrics["upload_numpy_array_bytes"] = float(len(wire))
        else:
            metrics.update(
                {
                    f"upload_{key}": float(value)
                    for key, value in payload_metrics(
                        arrays, int(config.get("server_round", 0)), 0, "uplink"
                    ).items()
                    if isinstance(value, (int, float))
                }
            )
        if self._global_arrays_before_fit is not None:
            squared = 0.0
            global_squared = 0.0
            for local, global_array in zip(arrays, self._global_arrays_before_fit, strict=True):
                delta = local.astype(np.float64, copy=False) - global_array.astype(
                    np.float64, copy=False
                )
                squared += float(np.sum(delta * delta))
                global_squared += float(
                    np.sum(global_array.astype(np.float64, copy=False) ** 2)
                )
            l2 = float(np.sqrt(squared))
            norm = float(np.sqrt(global_squared))
            metrics["client_update_l2"] = l2
            metrics["normalized_client_update_l2"] = l2 / norm if norm > 0 else 0.0
        metrics["strategy"] = self.strategy
        if self.strategy == "fedprox":
            metrics["fedprox_mu"] = float(self.fedprox_mu)
        self._proximal_global_parameters = None
        self._save_client_state(int(config.get("server_round", 0)))
        return upload_arrays, self.examples, metrics

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        self.set_model_parameters(parameters)
        self._evaluation_config = dict(config)
        try:
            metrics = self._finite(self.evaluate_fn())
        finally:
            self._evaluation_config = {}
        metrics["client_id"] = self.client_id
        return metrics["loss"], self.examples, metrics
