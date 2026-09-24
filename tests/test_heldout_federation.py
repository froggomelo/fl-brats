import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.common import ndarrays_to_parameters

from fl_brats.data.cohorts import read_split, validate_client_manifest, validate_split
from fl_brats.data.hashing import sha256_file
from fl_brats.federated.evaluation import HeldOutEvaluator
from fl_brats.federated.finalization import finalize_federation
from fl_brats.federated.parameters import state_to_arrays
from fl_brats.federated.revalidation import discover_checkpoints, evaluate_final_run, revalidate_run
from fl_brats.federated.strategy import create_research_fedavg


def fixture_run(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "resolved_config.yaml").write_text(
        json.dumps({"device": "cpu", "loss": {"name": "dice_ce"}})
    )
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "train_subjects": ["a"],
                "validation_subjects": ["b"],
                "test_subjects": ["c"],
                "dataset_index_path": "unused",
            }
        )
    )
    return run, split


@pytest.mark.parametrize(
    "left,right", [("train", "validation"), ("train", "test"), ("validation", "test")]
)
def test_split_overlap_is_rejected(left, right):
    split = {f"{s}_subjects": [s] for s in ("train", "validation", "test")}
    split[f"{right}_subjects"] = split[f"{left}_subjects"]
    with pytest.raises(ValueError, match="overlap"):
        validate_split(split)


def test_client_only_accepts_training_subjects(tmp_path):
    _, path = fixture_run(tmp_path)
    split = read_split(path)
    manifest = tmp_path / "client.json"
    for subject in ("b", "c", "unknown"):
        manifest.write_text(json.dumps({"subjects": [{"subject_id": subject}]}))
        with pytest.raises(ValueError, match="training subjects only"):
            validate_client_manifest(manifest, split)
    manifest.write_text(json.dumps({"subjects": [{"subject_id": "a", "split": "train"}]}))
    assert validate_client_manifest(manifest, split) == ["a"]


@pytest.mark.parametrize("algorithm", ["fedavg", "fedamm"])
def test_one_global_validation_selects_exact_checkpoint_and_stops(tmp_path, algorithm):
    model = torch.nn.Linear(2, 1)
    calls = []
    scores = iter([0.8, 0.7])

    def callback(number, arrays, config):
        saved = torch.load(tmp_path / f"checkpoints/round_{number:04d}.pt", weights_only=True)
        for array, tensor in zip(arrays, saved["model_state"].values(), strict=True):
            np.testing.assert_array_equal(array, tensor.numpy())
        calls.append(number)
        return 0.2, {"mean_dice": next(scores), "modality__all__mean_dice": 1.0}

    strategy = create_research_fedavg(
        {
            "num_clients": 1,
            "strategy": algorithm,
            "minimum_rounds": 2,
            "early_stopping_patience": 1,
        },
        run_dir=tmp_path,
        model=model,
        evaluate_fn=callback,
    )
    assert strategy.fraction_evaluate == strategy.min_evaluate_clients == 0
    assert strategy.configure_evaluate(1, None, None) == []
    assert strategy.evaluate(0, None) is None
    for number in (1, 2):
        arrays = [a + number for a in state_to_arrays(model)[1]]
        result = SimpleNamespace(
            parameters=ndarrays_to_parameters(arrays),
            num_examples=1,
            metrics={"mean_dice": 1.0, "wt_dice": 1.0},
        )
        parameters, _ = strategy.aggregate_fit(number, [(SimpleNamespace(cid="0"), result)], [])
        assert strategy.state.best_round is None if number == 1 else strategy.state.best_round == 1
        strategy.evaluate(number, parameters)
        assert strategy.evaluate(number, parameters) is None
    assert calls == [1, 2]
    assert strategy.state.best_round == 1
    assert strategy.state.best_metric == 0.8
    assert sha256_file(tmp_path / "checkpoints/best.pt") == sha256_file(
        tmp_path / "checkpoints/round_0001.pt"
    )
    assert strategy.configure_fit(3, None, None) == []
    assert strategy.state.status == "early_stopped"
    assert len((tmp_path / "metrics/validation_rounds.jsonl").read_text().splitlines()) == 2
    # Even an explicit diagnostic with a perfect score cannot change selection.
    diagnostic = SimpleNamespace(loss=0.0, num_examples=1, metrics={"mean_dice": 1.0})
    strategy.aggregate_evaluate(2, [(SimpleNamespace(cid="0"), diagnostic)], [])
    assert strategy.state.best_round == 1


def test_revalidation_resume_preserves_history_and_numeric_order(tmp_path, monkeypatch):
    run, split = fixture_run(tmp_path)
    for number in (10, 2, 1):
        torch.save(
            {
                "round": number,
                "model_config": {},
                "model_state": torch.nn.Linear(1, 1).state_dict(),
            },
            run / f"checkpoints/round_{number}.pt",
        )
    best = run / "checkpoints/best.pt"
    best.write_bytes(b"historical best")
    (run / "state.json").write_text('{"best_round": 10}')
    (run / "metrics").mkdir()
    historical = run / "metrics/evaluation_rounds.jsonl"
    historical.write_text('{"old": true}\n')
    before = (sha256_file(best), sha256_file(historical), sha256_file(run / "state.json"))
    assert [r for r, _ in discover_checkpoints(run)] == [1, 2, 10]
    calls = []
    monkeypatch.setattr(HeldOutEvaluator, "prepare", lambda self: None)
    monkeypatch.setattr(
        "fl_brats.federated.revalidation.create_model", lambda cfg: torch.nn.Linear(1, 1)
    )

    def evaluate(self, model, **kwargs):
        calls.append(1)
        assert self.subjects == ["b"]
        if len(calls) == 2:
            raise InterruptedError()
        return {"mean_dice": 0.7 if len(calls) == 1 else 0.9}, {}

    monkeypatch.setattr(HeldOutEvaluator, "run", evaluate)
    with pytest.raises(InterruptedError):
        revalidate_run(run, split)
    summary = revalidate_run(run, split)
    assert len(calls) == 4  # one success, interruption, two resumed successes
    assert summary["checkpoints_successfully_evaluated"] == 3
    assert summary["best_revalidated_round"] == 2  # stable earliest tie
    assert before == (sha256_file(best), sha256_file(historical), sha256_file(run / "state.json"))
    revalidate_run(run, split)
    assert len(calls) == 4
    split_data = read_split(split)
    split_data["validation_subjects"] = ["different"]
    split.write_text(json.dumps(split_data))
    with pytest.raises(ValueError, match="identity changed"):
        revalidate_run(run, split)


def test_dry_run_does_not_write(tmp_path):
    run, split = fixture_run(tmp_path)
    (run / "checkpoints/round_0001.pt").touch()
    result = revalidate_run(run, split, dry_run=True)
    assert result["total_checkpoints_discovered"] == 1
    assert not (run / "revalidation").exists()


def test_final_reporting_uses_test_only_and_explicit_final(tmp_path, monkeypatch):
    run, split = fixture_run(tmp_path)
    calls = []

    def report(self, checkpoint, destination):
        assert self.subjects == ["c"]
        assert self.metadata["selection_criterion"] is None
        with pytest.raises(ValueError, match="test evaluation"):
            self.callback(None, run)
        calls.append((checkpoint, destination))

    monkeypatch.setattr(HeldOutEvaluator, "evaluate_checkpoint", report)
    evaluate_final_run(run, split, checkpoint=Path("revalidation/best_revalidated.pt"))
    evaluate_final_run(run, split, final_round=150)
    assert calls[1][0] == run / "checkpoints/round_0150.pt"
    assert not (run / "state.json").exists()
    assert not (run / "metrics/validation_rounds.jsonl").exists()


def test_finalization_writes_completed_state_and_alias(tmp_path):
    model = torch.nn.Linear(1, 1)
    strategy = create_research_fedavg(
        {"num_clients": 1},
        run_dir=tmp_path,
        model=model,
        evaluate_fn=lambda *_: (0.1, {"mean_dice": 0.7}),
    )
    result = SimpleNamespace(
        parameters=ndarrays_to_parameters(state_to_arrays(model)[1]), num_examples=1, metrics={}
    )
    parameters, _ = strategy.aggregate_fit(1, [(SimpleNamespace(cid="0"), result)], [])
    strategy.evaluate(1, parameters)
    evaluator = SimpleNamespace(
        policy={"evaluate_best_checkpoint": False, "evaluate_final_checkpoint": False}, metadata={}
    )
    summary = finalize_federation(strategy, evaluator, {}, tmp_path, 1.0)
    assert summary["status"] == "completed"
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "completed"
    assert (tmp_path / "checkpoints/final.pt").read_bytes() == (
        tmp_path / "checkpoints/round_0001.pt"
    ).read_bytes()


def test_build_client_has_no_validation_loader(tmp_path, monkeypatch):
    from fl_brats.federated.orchestration import build_client

    run, split = fixture_run(tmp_path)
    manifest = tmp_path / "client_0.json"
    manifest.write_text(
        json.dumps(
            {
                "client_id": 0,
                "subjects": [
                    {
                        "subject_id": "a",
                        "paths": {
                            "t1": "/unused",
                            "t1ce": "/unused",
                            "t2": "/unused",
                            "flair": "/unused",
                            "seg": "/unused",
                        },
                    }
                ],
            }
        )
    )
    values = {"data": {"split_path": str(split)}, "device": "cpu"}
    monkeypatch.setattr(
        "fl_brats.federated.orchestration.create_model", lambda _: torch.nn.Linear(1, 1)
    )
    client = build_client(values, "client_0", tmp_path)
    assert client.validation_loader is None
    assert [r["subject_id"] for r in client.train_loader.dataset.data] == ["a"]
    with pytest.raises(RuntimeError, match="disabled"):
        client._evaluation_disabled()


def test_fixed_policy_across_training_seeds_and_expensive_cadence(tmp_path, monkeypatch):
    _, split = fixture_run(tmp_path)
    values = {"missing_modalities": {"enabled": True, "mode": "arbitrary"}, "device": "cpu"}
    first = HeldOutEvaluator({**values, "seed": 42}, split)
    second = HeldOutEvaluator({**values, "seed": 44}, split)
    assert first.metadata["natural_masks"] == second.metadata["natural_masks"]
    calls = []

    def evaluate(self, model, **kwargs):
        calls.append(kwargs)
        return {"loss": 0.0, "mean_dice": 0.5, "subjects": 1}, {}

    monkeypatch.setattr(HeldOutEvaluator, "run", evaluate)
    model = torch.nn.Linear(1, 1)
    callback = first.callback(model, tmp_path)
    arrays = state_to_arrays(model)[1]
    callback(1, arrays, {})
    callback(10, arrays, {})
    assert calls == [
        {"expensive": False, "robustness": False},
        {"expensive": True, "robustness": True},
    ]


def test_checkpoint_mutation_invalidates_revalidation_resume(tmp_path, monkeypatch):
    run, split = fixture_run(tmp_path)
    checkpoint = run / "checkpoints/round_0001.pt"
    torch.save(
        {"round": 1, "model_config": {}, "model_state": torch.nn.Linear(1, 1).state_dict()},
        checkpoint,
    )
    monkeypatch.setattr(HeldOutEvaluator, "prepare", lambda self: None)
    monkeypatch.setattr(HeldOutEvaluator, "run", lambda *args, **kwargs: ({"mean_dice": 0.5}, {}))
    monkeypatch.setattr(
        "fl_brats.federated.revalidation.create_model", lambda cfg: torch.nn.Linear(1, 1)
    )
    revalidate_run(run, split)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        revalidate_run(run, split)


@pytest.mark.parametrize("algorithm", ["fedavg", "fedamm"])
@pytest.mark.parametrize("transport", ["quantized", "sparse"])
def test_compressed_aggregation_retains_global_validation(tmp_path, algorithm, transport):
    from fl_brats.federated.model_client import ModelFedAvgClient

    model = torch.nn.Linear(2, 1)
    client_model = torch.nn.Linear(2, 1)

    def train():
        with torch.no_grad():
            for parameter in client_model.parameters():
                parameter.add_(0.1)
        return {"train_loss": 0.1}

    config = {"enabled": True, "mode": "fixed", "fixed": {"bit_width": 8}}
    if transport == "sparse":
        config = {"sparsification": {"enabled": True, "fraction": 1.0}}
    client = ModelFedAvgClient(client_model, train_fn=train, examples=1, quantization_config=config)
    calls = []

    def evaluate(number, arrays, config):
        calls.append(number)
        return 0.1, {"mean_dice": 0.5}

    strategy = create_research_fedavg(
        {"num_clients": 1, "strategy": algorithm},
        run_dir=tmp_path,
        model=model,
        evaluate_fn=evaluate,
        round_offset=3,
    )
    arrays, examples, metrics = client.fit(state_to_arrays(model)[1], strategy.on_fit_config_fn(1))
    result = SimpleNamespace(
        parameters=ndarrays_to_parameters(arrays), num_examples=examples, metrics=metrics
    )
    parameters, _ = strategy.aggregate_fit(1, [(SimpleNamespace(cid="0"), result)], [])
    strategy.evaluate(1, parameters)
    assert calls == [4]
    assert strategy.state.best_round == 4
    assert (tmp_path / "checkpoints/round_0004.pt").exists()


def test_corrected_resume_recovers_best_through_checkpoint_round(tmp_path):
    from fl_brats.federated.orchestration import load_resume_state
    from fl_brats.federated.research import CheckpointManager, ExperimentState

    run, split = fixture_run(tmp_path)
    values = {"data": {"split_path": str(split)}, "device": "cpu"}
    evaluator = HeldOutEvaluator(values)
    (run / "evaluation_metadata.json").write_text(json.dumps(evaluator.metadata))
    model = torch.nn.Linear(1, 1)
    manager = CheckpointManager(run, model, {})
    for number in (1, 2, 3):
        parameters = ndarrays_to_parameters([a + number for a in state_to_arrays(model)[1]])
        manager.save_round(number, parameters, ExperimentState(completed_round=number), {})
    manager.save_alias(run / "checkpoints/round_0003.pt", "best.pt")
    (run / "metrics").mkdir()
    (run / "metrics/validation_rounds.jsonl").write_text(
        "\n".join(json.dumps({"round": n, "mean_dice": n / 10}) for n in (1, 2, 3)) + "\n"
    )
    continuation = tmp_path / "continuation"
    state = load_resume_state(
        model, {**values, "resume": run / "checkpoints/round_0002.pt"}, continuation
    )
    assert state.best_round == 2
    assert state.best_metric == 0.2
    assert sha256_file(Path(state.best_checkpoint)) == sha256_file(
        run / "checkpoints/round_0002.pt"
    )


def test_distributed_server_accepts_slurm_bootstrap_and_wires_global_evaluation(
    tmp_path, monkeypatch
):
    from fl_brats.federated.server import start_server

    _, split = fixture_run(tmp_path)
    run = tmp_path / "distributed"
    run.mkdir()
    (run / "server_address.txt").write_text("localhost:8080")
    values = {
        "device": "cpu",
        "num_clients": 1,
        "num_rounds": 1,
        "data": {"split_path": str(split)},
        "model": {"name": "unet", "channels": [2, 4], "strides": [2]},
        "evaluation": {"evaluate_best_checkpoint": False, "evaluate_final_checkpoint": False},
    }
    monkeypatch.setattr(HeldOutEvaluator, "prepare", lambda self: None)
    monkeypatch.setattr(
        HeldOutEvaluator,
        "run",
        lambda *a, **k: ({"loss": 0.1, "mean_dice": 0.6, "subjects": 1}, {}),
    )

    def flower(**kwargs):
        strategy = kwargs["strategy"]
        result = SimpleNamespace(
            parameters=ndarrays_to_parameters(state_to_arrays(strategy.checkpoints.model)[1]),
            num_examples=1,
            metrics={},
        )
        parameters, _ = strategy.aggregate_fit(1, [(SimpleNamespace(cid="0"), result)], [])
        strategy.evaluate(1, parameters)
        assert strategy.configure_evaluate(1, parameters, None) == []

    monkeypatch.setattr("flwr.server.start_server", flower)
    start_server("localhost:8080", values, run_dir=run)
    assert json.loads((run / "state.json").read_text())["status"] == "completed"
    assert json.loads((run / "metrics/validation_rounds.jsonl").read_text())["mean_dice"] == 0.6
    before = sha256_file(run / "checkpoints/best.pt")
    with pytest.raises(ValueError, match="already contains"):
        start_server("localhost:8080", values, run_dir=run)
    assert sha256_file(run / "checkpoints/best.pt") == before
