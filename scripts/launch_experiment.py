#!/usr/bin/env python3
"""Launch YAML-defined experiments locally or through the generic SLURM job."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and list values."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and value and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_experiment_yaml(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Resolve relative ``extends`` without importing the training CLI."""
    resolved = path.expanduser().resolve()
    chain = set() if seen is None else set(seen)
    if resolved in chain:
        raise ValueError(f"cyclic experiment config inheritance at {resolved}")
    chain.add(resolved)
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"experiment config must contain a YAML mapping: {resolved}")
    parents = loaded.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
        raise ValueError(f"extends must be a path or list of paths: {resolved}")
    merged: dict[str, Any] = {}
    for parent in parents:
        merged = _deep_merge(
            merged,
            _load_experiment_yaml((resolved.parent / parent).resolve(), chain),
        )
    return _deep_merge(merged, loaded)


def _slug(value: str) -> str:
    return value.strip("/").replace("/", "__").replace(" ", "_")


def _command(config: Path, values: dict[str, Any], run_dir: Path, seed: int) -> list[str]:
    if values["experiment"]["family"] == "centralized":
        data, training = values["data"], values["training"]
        return [
            "fedbrats", "train-centralized",
            "--index-path", str(data["index_path"]),
            "--split-path", str(data["split_path"]),
            "--partition-path", str(data["partition_path"]),
            "--initial-weights", str(values["initial_weights"]),
            "--run-dir", str(run_dir),
            "--epochs", str(training["epochs"]),
            "--minimum-epochs", str(training.get("minimum_epochs", 1)),
            "--early-stopping-patience", str(training["early_stopping_patience"]),
            "--early-stopping-min-delta", str(training["early_stopping_min_delta"]),
            "--batch-size", str(training["batch_size"]),
            "--workers", str(training.get("num_workers", 0)),
            "--seed", str(seed), "--device", str(values.get("device", "auto")),
        ]
    return ["fedbrats", "train-federated", "--config", str(config), "--run-dir", str(run_dir)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=("local", "slurm"), default="slurm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    values = _load_experiment_yaml(args.config.resolve())
    experiment = values["experiment"]
    seeds = [args.seed] if args.seed is not None else list(experiment.get("seeds", [42]))
    family, variant = str(experiment["family"]), str(experiment.get("variant", args.config.stem))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    submitted: list[dict[str, Any]] = []
    for seed in seeds:
        run_id = f"{_slug(variant)}_seed_{seed}_{stamp}"
        output_root = Path(
            os.environ.get("FL_BRATS_OUTPUT_ROOT", values.get("output_root", "outputs"))
        )
        log_root = Path(os.environ.get("FL_BRATS_LOG_ROOT", values.get("log_root", "logs")))
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        if not log_root.is_absolute():
            log_root = REPO_ROOT / log_root
        run_dir = output_root / family / run_id
        log_dir = log_root / family / run_id
        runtime_config = log_dir / "resolved-launch-config.yaml"
        resolved = dict(values)
        weights_by_seed = resolved.get("initial_weights_by_seed", {})
        if seed in weights_by_seed or str(seed) in weights_by_seed:
            resolved["initial_weights"] = weights_by_seed.get(seed, weights_by_seed.get(str(seed)))
        resolved.update(
            seed=seed,
            run_name=run_id,
            output_root=str(run_dir.parent),
            log_root=str(log_dir),
        )
        command = _command(runtime_config, resolved, run_dir, seed)
        if args.dry_run:
            submitted.append({"run_id": run_id, "command": command, "config": resolved})
            continue
        log_dir.mkdir(parents=True, exist_ok=False)
        runtime_config.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        if args.mode == "local":
            with (log_dir / "launcher.out").open(
                "w", encoding="utf-8"
            ) as stdout, (log_dir / "launcher.err").open(
                "w", encoding="utf-8"
            ) as stderr:
                result = subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr)
            if result.returncode:
                return result.returncode
            submitted.append({"run_id": run_id, "run_dir": str(run_dir), "mode": "local"})
            continue
        slurm = resolved.get("slurm", {})
        use_apptainer = str(bool(slurm.get("use_apptainer", False))).lower()
        image = str(slurm.get("image", ""))
        sbatch = [
            "sbatch", "--parsable", f"--job-name=flb-{_slug(variant)[:40]}",
            f"--partition={slurm['partition']}", f"--nodes={slurm.get('nodes', 1)}",
            f"--ntasks={slurm.get('tasks', 1)}",
            f"--cpus-per-task={slurm.get('cpus_per_task', 8)}",
            f"--time={slurm.get('time', '72:00:00')}",
            f"--mem={slurm.get('memory', '64G')}",
        ]
        for key, flag in (("account", "--account"), ("qos", "--qos"), ("gres", "--gres")):
            if slurm.get(key):
                sbatch.append(f"{flag}={slurm[key]}")
        if slurm.get("exclusive"):
            sbatch.append("--exclusive")
        interconnect = resolved.get("communication_measurement", {}).get(
            "slurm_interconnect", {}
        )
        if interconnect.get("enabled"):
            sbatch.append(
                f"--acctg-freq=network={int(interconnect.get('sampling_seconds', 30))}"
            )
        if family != "centralized":
            clients = int(resolved.get("federated", {}).get("num_clients", 4))
            server_command = sbatch + [
                f"--output={log_dir}/server-%j.out", f"--error={log_dir}/server-%j.err",
                str(REPO_ROOT / "slurm" / "_shared" / "federated_server.sbatch"),
                str(runtime_config), str(run_dir),
                str(resolved.get("federated", {}).get("server_port", 8080)),
                use_apptainer, image,
            ]
            server_job = subprocess.check_output(server_command, cwd=REPO_ROOT, text=True).strip()
            client_command = sbatch + [
                f"--array=0-{clients - 1}",
                f"--output={log_dir}/client-%A-%a.out",
                f"--error={log_dir}/client-%A-%a.err",
                str(REPO_ROOT / "slurm" / "_shared" / "federated_client.sbatch"),
                str(runtime_config), str(run_dir), use_apptainer, image,
            ]
            client_job = subprocess.check_output(client_command, cwd=REPO_ROOT, text=True).strip()
            cleanup_job = subprocess.check_output(
                [
                    "sbatch", "--parsable", f"--partition={slurm['partition']}",
                    f"--dependency=afterany:{client_job}",
                    f"--output={log_dir}/cleanup-%j.out", f"--error={log_dir}/cleanup-%j.err",
                    f"--wrap=scancel {server_job} || true",
                ],
                cwd=REPO_ROOT, text=True,
            ).strip()
            jobs = {"server": server_job, "clients": client_job, "cleanup": cleanup_job}
            (log_dir / "slurm-job-ids.json").write_text(
                json.dumps(jobs, indent=2) + "\n", encoding="utf-8"
            )
            submitted.append({"run_id": run_id, "run_dir": str(run_dir), "jobs": jobs})
            continue
        sbatch.extend(
            [
                f"--output={log_dir}/slurm-%j.out",
                f"--error={log_dir}/slurm-%j.err",
                str(REPO_ROOT / "slurm" / "_shared" / "run_experiment.sbatch"),
                use_apptainer, image,
                *command,
            ]
        )
        job_id = subprocess.check_output(sbatch, cwd=REPO_ROOT, text=True).strip()
        (log_dir / "slurm-job-id.txt").write_text(job_id + "\n", encoding="utf-8")
        submitted.append({"run_id": run_id, "run_dir": str(run_dir), "job_id": job_id})
    print(json.dumps(submitted, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
