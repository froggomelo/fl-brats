"""Factory for standard and research-instrumented synchronous Flower FedAvg."""

import warnings
from typing import Any

from .evaluation import evaluation_policy


def create_fedavg(config: dict[str, Any], evaluate_fn: Any = None) -> Any:
    import flwr as fl

    return fl.server.strategy.FedAvg(
        fraction_fit=float(config.get("fraction_fit", 1.0)),
        fraction_evaluate=0.0,
        min_fit_clients=int(config.get("min_fit_clients", config.get("num_clients", 4))),
        min_evaluate_clients=0,
        min_available_clients=int(
            config.get("min_available_clients", config.get("num_clients", 4))
        ),
        accept_failures=bool(config.get("accept_failures", False)),
        evaluate_fn=evaluate_fn,
    )


def create_research_fedavg(
    config: dict[str, Any],
    *,
    run_dir: Any,
    model: Any,
    evaluate_fn: Any = None,
    initial_state: Any = None,
    round_offset: int = 0,
) -> Any:
    """Build the classic Flower strategy with server-owned experiment artifacts."""
    from flwr.common import ndarrays_to_parameters

    from .fedamm import FedAMMStrategy
    from .parameters import state_to_arrays
    from .research import (
        CheckpointManager,
        ResearchFedAvgStrategy,
        RoundArtifactWriter,
        weighted_average_metrics,
    )

    missing = config.get("missing_modalities", {})
    evaluation = evaluation_policy(config)
    diagnostics = bool(evaluation["client_diagnostics"])
    if config.get("fraction_evaluate", 0) and not diagnostics:
        warnings.warn(
            "fraction_evaluate is ignored: use evaluation.client_diagnostics for optional "
            "training-partition diagnostics; selection uses global validation.",
            UserWarning,
            stacklevel=2,
        )
    robustness_enabled = bool(missing.get("enabled", False)) and bool(
        evaluation.get("evaluate_all_non_empty_combinations", True)
    )
    robustness_interval = max(int(evaluation.get("robustness_every_rounds", 10)), 1)
    expensive_interval = max(
        int(evaluation.get("expensive_metrics_every_rounds", robustness_interval)), 1
    )
    training = config.get("training", {})
    target_rounds = int(
        training.get("max_rounds", config.get("max_rounds", config.get("num_rounds", 50)))
        if isinstance(training, dict)
        else config.get("max_rounds", config.get("num_rounds", 50))
    )

    def evaluation_config(server_round: int) -> dict[str, bool | int]:
        global_round = round_offset + int(server_round)
        final_round = global_round >= target_rounds
        return {
            "server_round": global_round,
            "evaluate_modality_robustness": robustness_enabled
            and (global_round % robustness_interval == 0 or final_round),
            "compute_expensive_metrics": robustness_enabled
            and (global_round % expensive_interval == 0 or final_round),
        }

    common = dict(
        client_diagnostics=diagnostics,
        writer=RoundArtifactWriter(run_dir),
        checkpoints=CheckpointManager(run_dir, model, dict(config.get("model", {}))),
        fraction_fit=float(config.get("fraction_fit", 1.0)),
        fraction_evaluate=float(config.get("fraction_evaluate", 1.0)) if diagnostics else 0.0,
        min_fit_clients=int(config.get("min_fit_clients", config.get("num_clients", 4))),
        min_evaluate_clients=int(config.get("min_evaluate_clients", 1)) if diagnostics else 0,
        min_available_clients=int(
            config.get("min_available_clients", config.get("num_clients", 4))
        ),
        accept_failures=bool(config.get("accept_failures", False)),
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=weighted_average_metrics,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
        on_evaluate_config_fn=evaluation_config,
        on_fit_config_fn=lambda server_round: {"server_round": round_offset + int(server_round)},
        initial_parameters=ndarrays_to_parameters(state_to_arrays(model)[1]),
        early_stopping_patience=config.get("early_stopping_patience"),
        minimum_rounds=int(config.get("minimum_rounds", 0)),
        initial_state=initial_state,
        round_offset=round_offset,
    )
    if config.get("strategy") == "fedamm":
        return FedAMMStrategy(
            model=model,
            clusters_per_combination=int(
                config.get("fedamm", {}).get("clusters_per_combination", 1)
            ),
            **common,
        )

    return ResearchFedAvgStrategy(
        **common,
    )
