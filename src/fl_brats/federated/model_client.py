"""Segmentation-aware Flower client used by local and distributed runs."""
# ruff: noqa: E501

from __future__ import annotations

import math
import time
from collections.abc import Sized
from typing import Any, Callable, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from fl_brats.data.modalities import MODALITIES, modality_counts
from fl_brats.training.engine import evaluate, train_local_epochs

from .client import FedAvgClient
from .communication import payload_metrics
from .parameters import load_arrays, state_keys, state_to_arrays


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
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.keys = keys or state_keys(model)
        self.train_loader, self.validation_loader = train_loader, validation_loader
        self.loss_fn, self.optimizer = loss_fn, optimizer
        self.local_epochs, self.client_id = local_epochs, client_id
        if examples is None:
            examples = len(cast(Sized, train_loader.dataset)) if train_loader is not None else 0
        if examples <= 0:
            raise ValueError(f"client {client_id!r} has an empty training partition")
        if train_fn is None:
            if train_loader is None or loss_fn is None or optimizer is None:
                raise ValueError("train_fn or train_loader, loss_fn, and optimizer are required")
            train_fn = self._train
        if evaluate_fn is None:
            if validation_loader is None or loss_fn is None:
                raise ValueError("evaluate_fn or validation_loader and loss_fn are required")
            evaluate_fn = self._evaluate
        super().__init__(self.model, self.keys, train_fn, evaluate_fn, examples)

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
        )
        result["local_training_seconds"] = time.monotonic() - started
        if self.train_loader is not None:
            result.update(modality_count_metrics(getattr(self.train_loader.dataset, "data", [])))
        return result

    def _evaluate(self) -> dict[str, float]:
        assert self.validation_loader is not None and self.loss_fn is not None
        return evaluate(self.model, self.validation_loader, self.loss_fn, self.device)

    @staticmethod
    def _finite(metrics: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in metrics.items():
            output[key] = value if isinstance(value, str) else float(value)
        numeric = [value for value in output.values() if isinstance(value, float)]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("client produced non-finite loss or metric")
        return output

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        self.set_model_parameters(parameters)
        if "fedamm_global_prototypes_json" in config:
            setattr(self.model, "global_prototypes_json", config["fedamm_global_prototypes_json"])
        metrics = self._finite(self.train_fn())
        arrays = self.get_model_parameters()
        metrics["client_id"] = self.client_id  # Flower scalar contract permits strings.
        metrics["download_payload_bytes"] = float(sum(item.nbytes for item in parameters))
        metrics.update(
            {
                f"upload_{key}": float(value)
                for key, value in payload_metrics(
                    arrays, int(config.get("server_round", 0)), 0, "uplink"
                ).items()
                if isinstance(value, (int, float))
            }
        )
        return arrays, self.examples, metrics

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        self.set_model_parameters(parameters)
        metrics = self._finite(self.evaluate_fn())
        return metrics["loss"], self.examples, metrics
