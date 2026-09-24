import json
from types import SimpleNamespace

import numpy as np
import torch
from flwr.common import ndarrays_to_parameters

from fl_brats.federated.fedamm import aggregate_fedamm_arrays
from fl_brats.federated.orchestration import load_resume_state, rounds_to_run
from fl_brats.federated.parameters import state_to_arrays
from fl_brats.federated.research import (
    CheckpointManager,
    ClientRoundRecord,
    ExperimentState,
    RoundArtifactWriter,
)
from fl_brats.federated.strategy import create_research_fedavg
from fl_brats.models.factory import create_model


def test_server_artifacts_and_checkpoint_are_atomic_and_named(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    writer = RoundArtifactWriter(tmp_path)
    state = ExperimentState(completed_round=1, uploaded_bytes=12)
    writer.write_client_records(
        [
            ClientRoundRecord(
                1,
                "client_0",
                2,
                0.3,
                0.2,
                0.6,
                0.5,
                0.4,
                0.5,
                1.0,
                12,
                10,
                0.25,
                0.05,
                "completed",
            )
        ]
    )
    writer.write_round_record({"round": 1})
    writer.write_communication_record({"round": 1, "uploaded_bytes": 12})
    writer.write_state(state)
    manager = CheckpointManager(tmp_path, model, {"name": "tiny"})
    _, arrays = state_to_arrays(model)
    checkpoint = manager.save_round(1, ndarrays_to_parameters(arrays), state, {"loss": 0.2})

    assert checkpoint.exists()
    assert json.loads((tmp_path / "state.json").read_text())["completed_round"] == 1
    assert len((tmp_path / "metrics" / "clients.jsonl").read_text().splitlines()) == 1
    assert torch.load(checkpoint, weights_only=False)["model_config"] == {"name": "tiny"}


def test_fedamm_aggregates_modality_encoders_by_modality_counts() -> None:
    keys = ["modality_encoders.t1.weight", "decoder.weight"]
    first = SimpleNamespace(
        parameters=ndarrays_to_parameters([np.array([2.0]), np.array([10.0])]),
        num_examples=1,
        metrics={"fedamm_modality_count_t1": 10.0},
    )
    second = SimpleNamespace(
        parameters=ndarrays_to_parameters([np.array([8.0]), np.array([20.0])]),
        num_examples=3,
        metrics={"fedamm_modality_count_t1": 0.0},
    )
    arrays = aggregate_fedamm_arrays(keys, [(None, first), (None, second)])
    assert arrays[0].tolist() == [2.0]
    assert arrays[1].tolist() == [17.5]


def test_fedamm_architecture_uses_plain_strategy_when_configured_as_fedavg(tmp_path) -> None:
    model = create_model(
        {
            "name": "fedamm",
            "channels": [2, 4],
            "out_channels": 3,
            "modality_weighted_aggregation": False,
        }
    )
    strategy = create_research_fedavg(
        {"strategy": "fedavg", "model": {"name": "fedamm"}, "num_clients": 1},
        run_dir=tmp_path,
        model=model,
    )
    assert strategy.__class__.__name__ == "ResearchFedAvgStrategy"


def test_resume_restores_state_history_and_global_round_offset(tmp_path) -> None:
    source = tmp_path / "source"
    continuation = tmp_path / "continuation"
    for root in (source, continuation):
        for relative in ("checkpoints", "metrics", "communication"):
            (root / relative).mkdir(parents=True, exist_ok=True)

    model = torch.nn.Linear(2, 1)
    state = ExperimentState(
        completed_round=3,
        best_round=2,
        best_metric=0.75,
        uploaded_bytes=30,
        downloaded_bytes=40,
    )
    manager = CheckpointManager(source, model, {"name": "tiny"})
    _, arrays = state_to_arrays(model)
    checkpoint = manager.save_round(
        3, ndarrays_to_parameters(arrays), state, {"round": 3}
    )
    best = manager.save_alias(checkpoint, "best.pt")
    state.latest_checkpoint = str(checkpoint)
    state.best_checkpoint = str(best)
    RoundArtifactWriter(source).write_state(state)
    (source / "metrics" / "rounds.jsonl").write_text(
        '{"round": 1}\n{"round": 3}\n{"round": 4}\n', encoding="utf-8"
    )

    restored_model = torch.nn.Linear(2, 1)
    restored = load_resume_state(
        restored_model, {"resume": str(checkpoint)}, continuation
    )

    assert restored is not None
    assert restored.completed_round == 3
    assert restored.best_round is None  # historical training-partition scores are not validation
    assert restored.status == "running"
    assert restored.best_checkpoint is None
    assert rounds_to_run({"num_rounds": 5}, restored) == 2
    assert [
        json.loads(line)["round"]
        for line in (continuation / "metrics" / "rounds.jsonl").read_text().splitlines()
    ] == [1, 3]

    strategy = create_research_fedavg(
        {"num_clients": 1},
        run_dir=continuation,
        model=restored_model,
        initial_state=restored,
        round_offset=restored.completed_round,
    )
    assert strategy.global_round(1) == 4


def test_resume_rejects_checkpoint_at_configured_final_round(tmp_path) -> None:
    state = ExperimentState(completed_round=5)
    try:
        rounds_to_run({"num_rounds": 5}, state)
    except ValueError as error:
        assert "already completed 5 of 5 rounds" in str(error)
    else:
        raise AssertionError("completed checkpoint should not start another Flower server")


def test_resumed_aggregation_writes_global_round_number(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    strategy = create_research_fedavg(
        {"num_clients": 1},
        run_dir=tmp_path,
        model=model,
        initial_state=ExperimentState(completed_round=3),
        round_offset=3,
    )
    _, arrays = state_to_arrays(model)
    strategy._round_started[1] = 0.0
    result = SimpleNamespace(
        parameters=ndarrays_to_parameters(arrays),
        num_examples=2,
        metrics={
            "client_id": "client_0",
            "train_loss": 0.2,
            "loss": 0.1,
            "mean_dice": 0.8,
            "upload_numpy_array_bytes": 10,
            "download_payload_bytes": 20,
        },
    )

    parameters, _ = strategy.aggregate_fit(1, [(SimpleNamespace(cid="0"), result)], [])

    assert parameters is not None
    assert strategy.state.completed_round == 4
    assert (tmp_path / "checkpoints" / "round_0004.pt").exists()
    strategy.evaluate(1, parameters)
    record = json.loads((tmp_path / "metrics" / "rounds.jsonl").read_text().splitlines()[-1])
    assert record["round"] == 4


def test_fedamm_evaluation_schedule_is_lightweight_except_every_tenth_and_final(
    tmp_path,
) -> None:
    model = torch.nn.Linear(2, 1)
    strategy = create_research_fedavg(
        {
            "num_clients": 1,
            "num_rounds": 25,
            "missing_modalities": {
                "enabled": True,
                "evaluation": {
                    "evaluate_all_non_empty_combinations": True,
                    "robustness_every_rounds": 10,
                    "expensive_metrics_every_rounds": 10,
                },
            },
        },
        run_dir=tmp_path,
        model=model,
    )

    assert strategy.on_evaluate_config_fn(1) == {
        "server_round": 1,
        "evaluate_modality_robustness": False,
        "compute_expensive_metrics": False,
    }
    assert strategy.on_evaluate_config_fn(10)["evaluate_modality_robustness"] is True
    assert strategy.on_evaluate_config_fn(10)["compute_expensive_metrics"] is True
    assert strategy.on_evaluate_config_fn(25)["evaluate_modality_robustness"] is True
