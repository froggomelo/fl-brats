"""Factory for standard and research-instrumented synchronous Flower FedAvg."""

from typing import Any


def create_fedavg(config: dict[str, Any], evaluate_fn: Any = None) -> Any:
    import flwr as fl

    return fl.server.strategy.FedAvg(
        fraction_fit=float(config.get("fraction_fit", 1.0)),
        fraction_evaluate=float(config.get("fraction_evaluate", 1.0)),
        min_fit_clients=int(config.get("min_fit_clients", config.get("num_clients", 4))),
        min_evaluate_clients=int(config.get("min_evaluate_clients", config.get("num_clients", 4))),
        min_available_clients=int(
            config.get("min_available_clients", config.get("num_clients", 4))
        ),
        accept_failures=bool(config.get("accept_failures", False)),
        evaluate_fn=evaluate_fn,
    )


def create_research_fedavg(
    config: dict[str, Any], *, run_dir: Any, model: Any, evaluate_fn: Any = None
) -> Any:
    """Build the classic Flower strategy with server-owned experiment artifacts."""
    from .fedamm import FedAMMStrategy
    from .research import (
        CheckpointManager,
        ResearchFedAvgStrategy,
        RoundArtifactWriter,
        weighted_average_metrics,
    )

    common = dict(
        writer=RoundArtifactWriter(run_dir),
        checkpoints=CheckpointManager(run_dir, model, dict(config.get("model", {}))),
        fraction_fit=float(config.get("fraction_fit", 1.0)),
        fraction_evaluate=float(config.get("fraction_evaluate", 1.0)),
        min_fit_clients=int(config.get("min_fit_clients", config.get("num_clients", 4))),
        min_evaluate_clients=int(config.get("min_evaluate_clients", config.get("num_clients", 4))),
        min_available_clients=int(
            config.get("min_available_clients", config.get("num_clients", 4))
        ),
        accept_failures=bool(config.get("accept_failures", False)),
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=weighted_average_metrics,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )
    if config.get("strategy") == "fedamm":
        return FedAMMStrategy(
            model=model,
            clusters_per_combination=int(config.get("fedamm", {}).get("clusters_per_combination", 1)),
            **common,
        )

    return ResearchFedAvgStrategy(
        **common,
    )
