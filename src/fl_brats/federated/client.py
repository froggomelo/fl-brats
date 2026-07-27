"""A minimal Flower NumPyClient delegating to the shared engine."""

from typing import Any, Callable

import flwr as fl

from .parameters import load_arrays, state_to_arrays


class FedAvgClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: Any,
        keys: list[str],
        train_fn: Callable[[], dict[str, float]],
        evaluate_fn: Callable[[], dict[str, float]],
        examples: int,
    ) -> None:
        self.model, self.keys, self.train_fn, self.evaluate_fn, self.examples = (
            model,
            keys,
            train_fn,
            evaluate_fn,
            examples,
        )

    def get_parameters(self, config: dict[str, Any]) -> list[Any]:
        return state_to_arrays(self.model)[1]

    def fit(  # type: ignore[override]
        self, parameters: list[Any], config: dict[str, Any]
    ) -> tuple[list[Any], int, dict[str, float]]:
        load_arrays(self.model, self.keys, parameters)
        metrics = self.train_fn()
        return state_to_arrays(self.model)[1], self.examples, metrics

    def evaluate(  # type: ignore[override]
        self, parameters: list[Any], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, float]]:
        load_arrays(self.model, self.keys, parameters)
        metrics = self.evaluate_fn()
        return metrics["loss"], self.examples, metrics
