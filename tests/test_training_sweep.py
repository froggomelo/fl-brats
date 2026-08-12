import json

from fl_brats.federated.sweep import (
    DEFAULT_SWEEP,
    SweepRun,
    generate_fedavg_full_factorial_runs,
    generate_sweep_runs,
    summarize_run,
    write_fedavg_sweep_manifest,
    write_sweep_summary,
)


def test_default_sweep_generation_counts() -> None:
    runs = generate_sweep_runs(DEFAULT_SWEEP)

    assert sum(run.strategy == "fedavg" for run in runs) == 12
    assert sum(run.strategy == "fedprox" for run in runs) == 9


def test_sweep_run_id_is_deterministic() -> None:
    first = SweepRun("fedprox", local_epochs=10, batch_size=8, seed=42, mu=0.01)
    second = SweepRun("fedprox", local_epochs=10, batch_size=8, seed=42, mu=0.01)

    assert first.run_id == second.run_id == "E10_B08_mu0p01_seed42"


def test_fedavg_full_factorial_generates_360_unique_runs(tmp_path) -> None:
    runs = generate_fedavg_full_factorial_runs({}, tmp_path / "fedavg_sweep")
    rows = [run.manifest_row() for run in runs]

    assert len(rows) == 360
    assert len({row["run_id"] for row in rows}) == 360
    assert {row["alpha"] for row in rows} == {0.001, 0.1, 1.0}
    assert {
        "fedavg_K3_E1_B4_C1_skew-moderate_alpha0p001",
        "fedavg_K3_E1_B4_C1_skew-moderate_alpha0p1",
        "fedavg_K8_E20_B16_C1_skew-even_alpha1",
    }.issubset({row["run_id"] for row in rows})


def test_fedavg_full_factorial_manifest_is_written(tmp_path) -> None:
    sweep_root = tmp_path / "fedavg_sweep"
    runs = generate_fedavg_full_factorial_runs({}, sweep_root)

    rows = write_fedavg_sweep_manifest(sweep_root, runs)

    assert len(rows) == 360
    assert (sweep_root / "sweep_manifest.csv").exists()
    assert json.loads((sweep_root / "sweep_manifest.json").read_text())[0]["status"] == "pending"


def test_sweep_summary_combines_synthetic_outputs(tmp_path) -> None:
    run_dir = tmp_path / "training_sweep" / "fedavg" / "E01_B04_seed42"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "communication").mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "E01_B04_seed42",
                "status": "completed",
                "strategy": "fedavg",
                "seed": 42,
                "local_epochs": 1,
                "batch_size": 4,
                "actual_rounds": 2,
                "best_validation_dice": 0.8,
                "best_round": 2,
                "final_validation_dice": 0.75,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics" / "evaluation_rounds.jsonl").write_text(
        '{"round": 1, "validation_mean_dice": 0.6}\n'
        '{"round": 2, "validation_mean_dice": 0.8}\n',
        encoding="utf-8",
    )
    (run_dir / "metrics" / "clients.jsonl").write_text(
        '{"client_update_l2": 2.0}\n{"client_update_l2": 4.0}\n',
        encoding="utf-8",
    )
    (run_dir / "communication" / "summary.json").write_text(
        json.dumps({"total_wall_clock_time": 12.0, "total_bytes_transferred": 34}),
        encoding="utf-8",
    )

    row = summarize_run(run_dir)
    write_sweep_summary(tmp_path / "training_sweep", [row])

    assert row["round_to_90"] == 2
    assert row["mean_client_drift"] == 3.0
    assert (tmp_path / "training_sweep" / "training_sweep_summary.csv").exists()
    assert (tmp_path / "training_sweep" / "training_sweep_summary.json").exists()
