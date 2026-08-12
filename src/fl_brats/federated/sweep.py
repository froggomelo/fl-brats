"""Training-configuration sweep orchestration for federated U-Net experiments."""

from __future__ import annotations

import csv
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fl_brats.federated.orchestration import run_local_federation
from fl_brats.utils.reproducibility import seed_everything


DEFAULT_SWEEP: dict[str, Any] = {
    "name": "training_sweep",
    "seeds": [42],
    "fedavg": {"local_epochs": [1, 5, 10, 20], "batch_size": [4, 8, 16]},
    "fedprox": {"local_epochs": [5, 10, 20], "batch_size": [4, 8, 16]},
}


@dataclass(frozen=True)
class SweepRun:
    strategy: str
    local_epochs: int
    batch_size: int
    seed: int
    max_rounds: int | None = None
    mu: float | None = None

    @property
    def run_id(self) -> str:
        parts = [
            f"E{self.local_epochs:02d}",
            f"B{self.batch_size:02d}",
        ]
        if self.strategy == "fedprox":
            parts.append(f"mu{_slug_float(self.mu if self.mu is not None else 0.0)}")
        parts.append(f"seed{self.seed}")
        return "_".join(parts)


@dataclass(frozen=True)
class FedAvgFullFactorialRun:
    num_clients: int
    local_epochs: int
    batch_size: int
    client_fraction: float
    data_skew: str
    alpha: float
    partition_seed: int
    model_seed: int
    training_seed: int
    partition_manifest: str
    output_dir: str
    max_rounds: int | None = None

    @property
    def run_id(self) -> str:
        c_text = _slug_float(self.client_fraction)
        alpha_text = _slug_float(self.alpha)
        return (
            f"fedavg_K{self.num_clients}_E{self.local_epochs}_B{self.batch_size}"
            f"_C{c_text}_skew-{self.data_skew}_alpha{alpha_text}"
        )

    def manifest_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "K": self.num_clients,
            "E": self.local_epochs,
            "B": self.batch_size,
            "C": self.client_fraction,
            "data_skew": self.data_skew,
            "alpha": self.alpha,
            "partition_seed": self.partition_seed,
            "model_seed": self.model_seed,
            "training_seed": self.training_seed,
            "number_of_rounds": self.max_rounds,
            "partition_manifest": self.partition_manifest,
            "status": "pending",
            "output_dir": self.output_dir,
        }


def _slug_float(value: float) -> str:
    text = f"{value:g}"
    return re.sub(r"[^0-9A-Za-z]+", "p", text).strip("p") or "0"


def load_sweep_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_SWEEP)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sweep config must be a YAML mapping")
    merged = dict(DEFAULT_SWEEP)
    merged.update(payload)
    return merged


def generate_sweep_runs(config: dict[str, Any]) -> list[SweepRun]:
    seeds = [int(seed) for seed in config.get("seeds", [42])]
    max_rounds = config.get("max_rounds")
    training = config.get("training", {})
    if max_rounds is None and isinstance(training, dict):
        max_rounds = training.get("max_rounds")
    runs: list[SweepRun] = []
    for strategy in ("fedavg", "fedprox"):
        section = config.get(strategy, {})
        if not section:
            continue
        local_epochs = [int(value) for value in section.get("local_epochs", [])]
        batch_sizes = [int(value) for value in section.get("batch_size", [])]
        mu_values = [None]
        if strategy == "fedprox":
            fedprox = config.get("fedprox_config", config.get("fedprox_settings", {}))
            mu = section.get("mu", fedprox.get("mu", config.get("fedprox", {}).get("mu", 0.01)))
            mu_values = [float(value) for value in mu] if isinstance(mu, list) else [float(mu)]
        for seed, epochs, batch_size, mu in itertools.product(
            seeds, local_epochs, batch_sizes, mu_values
        ):
            runs.append(
                SweepRun(
                    strategy=strategy,
                    local_epochs=epochs,
                    batch_size=batch_size,
                    seed=seed,
                    max_rounds=int(max_rounds) if max_rounds is not None else None,
                    mu=mu,
                )
            )
    return runs


def _full_factorial_section(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("fedavg_full_factorial", {}))


def _partition_manifest_path(
    root: Path, *, k: int, data_skew: str, alpha: float, seed: int
) -> Path:
    return (
        root
        / f"K{k}"
        / f"data_skew-{data_skew}"
        / f"alpha{_slug_float(alpha)}"
        / f"seed{seed}"
        / "manifest.json"
    )


def generate_fedavg_full_factorial_runs(
    config: dict[str, Any], sweep_root: Path
) -> list[FedAvgFullFactorialRun]:
    section = _full_factorial_section(config)
    client_counts = [int(value) for value in section.get("client_counts", [3, 4, 5, 6, 8])]
    local_epochs = [int(value) for value in section.get("local_epochs", [1, 5, 10, 20])]
    batch_sizes = [int(value) for value in section.get("batch_size", [4, 8, 16])]
    client_fraction = float(section.get("client_fraction", 1.0))
    data_skews = [str(value) for value in section.get("data_skew", ["moderate", "even"])]
    alphas = [float(value) for value in section.get("alpha", [0.001, 0.1, 1.0])]
    partition_seed = int(section.get("partition_seed", config.get("partition_seed", 42)))
    model_seed = int(section.get("model_seed", config.get("model_seed", config.get("seed", 42))))
    training_seed = int(
        section.get("training_seed", config.get("training_seed", config.get("seed", 42)))
    )
    max_rounds = section.get("max_rounds", config.get("max_rounds"))
    training = config.get("training", {})
    if max_rounds is None and isinstance(training, dict):
        max_rounds = training.get("max_rounds")
    partition_root = Path(section.get("partition_root", sweep_root / "partitions"))
    runs: list[FedAvgFullFactorialRun] = []
    for k, epochs, batch_size, data_skew, alpha in itertools.product(
        client_counts, local_epochs, batch_sizes, data_skews, alphas
    ):
        manifest = _partition_manifest_path(
            partition_root, k=k, data_skew=data_skew, alpha=alpha, seed=partition_seed
        )
        output_dir = sweep_root / "fedavg" / (
            f"fedavg_K{k}_E{epochs}_B{batch_size}_C{_slug_float(client_fraction)}"
            f"_skew-{data_skew}_alpha{_slug_float(alpha)}"
        )
        runs.append(
            FedAvgFullFactorialRun(
                num_clients=k,
                local_epochs=epochs,
                batch_size=batch_size,
                client_fraction=client_fraction,
                data_skew=data_skew,
                alpha=alpha,
                partition_seed=partition_seed,
                model_seed=model_seed,
                training_seed=training_seed,
                partition_manifest=str(manifest),
                output_dir=str(output_dir),
                max_rounds=int(max_rounds) if max_rounds is not None else None,
            )
        )
    validate_fedavg_full_factorial_runs(runs)
    return runs


def validate_fedavg_full_factorial_runs(runs: list[FedAvgFullFactorialRun]) -> None:
    expected = 5 * 4 * 3 * 1 * 2 * 3
    rows = [run.manifest_row() for run in runs]
    run_ids = [row["run_id"] for row in rows]
    combos = [
        (row["K"], row["E"], row["B"], row["C"], row["data_skew"], row["alpha"])
        for row in rows
    ]
    if len(rows) != expected:
        raise ValueError(f"expected {expected} FedAvg configurations, got {len(rows)}")
    if len(set(run_ids)) != expected:
        raise ValueError("FedAvg sweep run IDs are not unique")
    if len(set(combos)) != expected:
        raise ValueError("FedAvg sweep combinations are not unique")
    if sorted({row["alpha"] for row in rows}) != [0.001, 0.1, 1.0]:
        raise ValueError("FedAvg sweep alpha values are not represented exactly")


def write_fedavg_sweep_manifest(
    sweep_root: Path, runs: list[FedAvgFullFactorialRun]
) -> list[dict[str, Any]]:
    sweep_root.mkdir(parents=True, exist_ok=True)
    rows = [run.manifest_row() for run in runs]
    (sweep_root / "sweep_manifest.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    columns = [
        "run_id",
        "K",
        "E",
        "B",
        "C",
        "data_skew",
        "alpha",
        "partition_seed",
        "model_seed",
        "training_seed",
        "number_of_rounds",
        "partition_manifest",
        "status",
        "output_dir",
    ]
    with (sweep_root / "sweep_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def values_for_run(base_values: dict[str, Any], run: SweepRun) -> dict[str, Any]:
    values = dict(base_values)
    values["seed"] = run.seed
    values["strategy"] = run.strategy
    values["local_epochs"] = run.local_epochs
    values["batch_size"] = run.batch_size
    values["local_batch_size"] = run.batch_size
    training = dict(values.get("training", {}))
    training["batch_size"] = run.batch_size
    if run.max_rounds is not None:
        training["max_rounds"] = run.max_rounds
        values["max_rounds"] = run.max_rounds
        values["num_rounds"] = run.max_rounds
    values["training"] = training
    federated = dict(values.get("federated", {}))
    federated["strategy"] = run.strategy
    values["federated"] = federated
    if run.strategy == "fedprox":
        fedprox = dict(values.get("fedprox", {}))
        fedprox["mu"] = float(run.mu if run.mu is not None else 0.0)
        values["fedprox"] = fedprox
    return values


def values_for_fedavg_full_factorial_run(
    base_values: dict[str, Any], run: FedAvgFullFactorialRun
) -> dict[str, Any]:
    values = dict(base_values)
    values["seed"] = run.training_seed
    values["partition_seed"] = run.partition_seed
    values["model_seed"] = run.model_seed
    values["training_seed"] = run.training_seed
    values["strategy"] = "fedavg"
    values["num_clients"] = run.num_clients
    values["local_epochs"] = run.local_epochs
    values["batch_size"] = run.batch_size
    values["local_batch_size"] = run.batch_size
    values["fraction_fit"] = run.client_fraction
    values["fraction_evaluate"] = run.client_fraction
    values["min_fit_clients"] = run.num_clients
    values["min_evaluate_clients"] = run.num_clients
    values["min_available_clients"] = run.num_clients
    values["data_skew"] = run.data_skew
    values["modality_availability_alpha"] = run.alpha
    values["partition_manifest"] = run.partition_manifest
    values["partition_root"] = str(Path(run.partition_manifest).parent)
    data = dict(values.get("data", {}))
    data["partition_root"] = values["partition_root"]
    values["data"] = data
    training = dict(values.get("training", {}))
    training["batch_size"] = run.batch_size
    if run.max_rounds is not None:
        training["max_rounds"] = run.max_rounds
        values["max_rounds"] = run.max_rounds
        values["num_rounds"] = run.max_rounds
    values["training"] = training
    federated = dict(values.get("federated", {}))
    federated["strategy"] = "fedavg"
    federated["fraction_fit"] = run.client_fraction
    federated["fraction_evaluate"] = run.client_fraction
    values["federated"] = federated
    model = dict(values.get("model", {}))
    model.setdefault("concat_mask", True)
    values["model"] = model
    missing = dict(values.get("missing_modalities", {}))
    missing["enabled"] = True
    missing["mode"] = "arbitrary"
    missing["seed"] = run.partition_seed
    missing["availability_alpha"] = run.alpha
    values["missing_modalities"] = missing
    return values


def run_sweep(
    base_values: dict[str, Any],
    sweep_config: dict[str, Any],
    output_root: Path,
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    sweep_name = str(sweep_config.get("name", "training_sweep"))
    sweep_root = output_root / sweep_name
    summaries: list[dict[str, Any]] = []
    for run in generate_sweep_runs(sweep_config):
        run_dir = sweep_root / run.strategy / run.run_id
        summary_path = run_dir / "summary.json"
        if summary_path.exists() and not force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "completed":
                summaries.append(summarize_run(run_dir))
                continue
            raise RuntimeError(
                f"interrupted run requires restart or --force because resume is unavailable: {run_dir}"
            )
        if run_dir.exists() and force:
            import shutil

            shutil.rmtree(run_dir)
        values = values_for_run(base_values, run)
        seed_everything(
            int(values.get("seed", 42)), deterministic=bool(values.get("deterministic", True))
        )
        run_local_federation(values, run_dir)
        summaries.append(summarize_run(run_dir))
    write_sweep_summary(sweep_root, summaries)
    return summaries


def run_fedavg_full_factorial_sweep(
    base_values: dict[str, Any],
    sweep_config: dict[str, Any],
    output_root: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> list[dict[str, Any]]:
    sweep_root = output_root / str(
        _full_factorial_section(sweep_config).get("name", "fedavg_sweep")
    )
    runs = generate_fedavg_full_factorial_runs(sweep_config, sweep_root)
    rows = write_fedavg_sweep_manifest(sweep_root, runs)
    if dry_run:
        return rows
    summaries: list[dict[str, Any]] = []
    for run in runs:
        run_dir = Path(run.output_dir)
        summary_path = run_dir / "summary.json"
        if summary_path.exists() and not force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "completed":
                summaries.append(summarize_run(run_dir))
                continue
            raise RuntimeError(
                f"interrupted run requires restart or --force because resume is unavailable: {run_dir}"
            )
        if run_dir.exists() and force:
            import shutil

            shutil.rmtree(run_dir)
        if not Path(run.partition_manifest).exists():
            raise ValueError(
                f"partition manifest is missing for {run.run_id}: {run.partition_manifest}"
            )
        values = values_for_fedavg_full_factorial_run(base_values, run)
        seed_everything(run.training_seed, deterministic=bool(values.get("deterministic", True)))
        summary = run_local_federation(values, run_dir)
        summary.update(
            {
                "K": run.num_clients,
                "E": run.local_epochs,
                "B": run.batch_size,
                "C": run.client_fraction,
                "data_skew": run.data_skew,
                "alpha": run.alpha,
                "partition_seed": run.partition_seed,
                "model_seed": run.model_seed,
                "training_seed": run.training_seed,
                "partition_manifest": run.partition_manifest,
            }
        )
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summarize_run(run_dir))
    write_sweep_summary(sweep_root, summaries)
    return summaries


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _first_round_at(records: list[dict[str, Any]], target: float) -> int | None:
    for record in records:
        value = record.get("validation_mean_dice")
        if isinstance(value, (int, float)) and float(value) >= target:
            return int(record["round"])
    return None


def summarize_run(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    evaluations = _jsonl(run_dir / "metrics" / "evaluation_rounds.jsonl")
    rounds = _jsonl(run_dir / "metrics" / "rounds.jsonl")
    clients = _jsonl(run_dir / "metrics" / "clients.jsonl")
    communication = json.loads((run_dir / "communication" / "summary.json").read_text()) if (
        run_dir / "communication" / "summary.json"
    ).exists() else summary.get("efficiency", {})
    best = summary.get("best_validation_dice") or summary.get("best_validation_mean_dice")
    final = summary.get("final_validation_dice")
    if final is None and evaluations:
        final = evaluations[-1].get("validation_mean_dice")
    target_base = best if isinstance(best, (int, float)) else final
    drift = [
        float(record["client_update_l2"])
        for record in clients
        if isinstance(record.get("client_update_l2"), (int, float))
    ]
    return {
        "run_id": summary.get("run_id", run_dir.name),
        "strategy": summary.get("strategy", run_dir.parent.name),
        "seed": summary.get("seed"),
        "local_epochs": summary.get("local_epochs"),
        "batch_size": summary.get("batch_size"),
        "mu": summary.get("fedprox_mu"),
        "rounds_completed": summary.get("actual_rounds", summary.get("completed_rounds")),
        "best_val_dice": best,
        "best_round": summary.get("best_round"),
        "final_validation_dice": final,
        "test_dice": summary.get("test_mean_dice"),
        "round_to_90": _first_round_at(evaluations, 0.90 * float(target_base))
        if isinstance(target_base, (int, float))
        else None,
        "round_to_95": _first_round_at(evaluations, 0.95 * float(target_base))
        if isinstance(target_base, (int, float))
        else None,
        "total_training_seconds": communication.get("total_wall_clock_time"),
        "total_communication_bytes": communication.get("total_bytes_transferred"),
        "mean_client_drift": sum(drift) / len(drift) if drift else None,
        "max_client_drift": max(drift) if drift else None,
        "round_wall_seconds_mean": (
            sum(float(record.get("round_wall_seconds", 0.0)) for record in rounds) / len(rounds)
            if rounds
            else None
        ),
    }


def write_sweep_summary(sweep_root: Path, rows: list[dict[str, Any]]) -> None:
    sweep_root.mkdir(parents=True, exist_ok=True)
    (sweep_root / "training_sweep_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    columns = sorted({key for row in rows for key in row})
    with (sweep_root / "training_sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
