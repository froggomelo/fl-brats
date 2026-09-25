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
        # Unset index/split paths come from .env inside the job (DATASET_INDEX_PATH,
        # FL_BRATS_SPLIT_PATH), exactly as for federated runs.
        paths = [
            item
            for key, flag in (("index_path", "--index-path"), ("split_path", "--split-path"))
            if data.get(key)
            for item in (flag, str(data[key]))
        ]
        return [
            "fedbrats", "train-centralized", *paths,
            "--partition-path", str(data["partition_path"]),
            "--initial-weights", str(values["initial_weights"]),
            "--run-dir", str(run_dir),
            "--epochs", str(training["epochs"]),
            "--minimum-epochs", str(training.get("minimum_epochs", 1)),
            "--early-stopping-patience", str(training["early_stopping_patience"]),
            "--early-stopping-min-delta", str(training["early_stopping_min_delta"]),
            "--batch-size", str(training["batch_size"]),
            "--workers", str(training.get("num_workers", 0)),
            "--cache-mode", str(data.get("cache_mode", "dataset")),
            "--cache-location", str(data.get("cache_location", "shared")),
            "--pin-memory" if training.get("pin_memory") else "--no-pin-memory",
            "--seed", str(seed), "--device", str(values.get("device", "auto")),
            *(["--resume-checkpoint", str(values["resume"])] if values.get("resume") else []),
        ]
    return ["fedbrats", "train-federated", "--config", str(config), "--run-dir", str(run_dir)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=("local", "slurm"), default="slurm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--partition")
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Continue from a federated round checkpoint up to the configured total rounds.",
    )
    parser.add_argument(
        "--afterany",
        metavar="JOBID",
        help="Wait for this Slurm job before starting; its exit status is ignored.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config key with a YAML value, e.g. slurm.exclusive=true.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="NODES",
        help="Comma-separated nodes to keep this run off, added to slurm.exclude.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help=(
            "Skip the job that cancels a server stuck after its clients end; for callers "
            "that watch the run themselves (scripts/monitoring/campaign.py). It would otherwise hold "
            "a job slot for the whole finalization."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.resume_checkpoint and not args.resume_checkpoint.expanduser().is_file():
        parser.error(f"resume checkpoint does not exist: {args.resume_checkpoint}")
    values = _load_experiment_yaml(args.config.resolve())
    for assignment in args.set:
        key, separator, raw = assignment.partition("=")
        if not separator or not key:
            parser.error(f"--set expects KEY=VALUE, got {assignment!r}")
        destination = values
        *parents, leaf = key.split(".")
        for part in parents:
            destination = destination.setdefault(part, {})
        destination[leaf] = yaml.safe_load(raw)
    if args.exclude:
        slurm_values = values.setdefault("slurm", {})
        configured = slurm_values.get("exclude") or []
        configured = configured.split(",") if isinstance(configured, str) else list(configured)
        extra = [node for item in args.exclude for node in item.split(",") if node]
        slurm_values["exclude"] = sorted(set(configured) | set(extra))
    experiment = values["experiment"]
    seeds = [args.seed] if args.seed is not None else list(experiment.get("seeds", [42]))
    family, variant = str(experiment["family"]), str(experiment.get("variant", args.config.stem))
    # Include microseconds so rapid retries cannot reuse a partially created
    # run directory from an earlier invocation in the same second.
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    submitted: list[dict[str, Any]] = []
    previous_job = args.afterany
    for seed in seeds:
        base_run_id = f"{_slug(variant)}_seed_{seed}_{stamp}"
        run_id = base_run_id
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
        if not args.dry_run:
            suffix = 1
            while True:
                try:
                    log_dir.mkdir(parents=True, exist_ok=False)
                    break
                except FileExistsError:
                    run_id = f"{base_run_id}_{suffix}"
                    suffix += 1
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
        if args.resume_checkpoint:
            resolved["resume"] = str(args.resume_checkpoint.expanduser().resolve())
        slurm_overrides = dict(resolved.get("slurm", {}))
        if args.partition:
            slurm_overrides["partition"] = args.partition
        if args.image:
            slurm_overrides["image"] = str(args.image)
        resolved["slurm"] = slurm_overrides
        command = _command(runtime_config, resolved, run_dir, seed)
        if args.dry_run:
            submitted.append({"run_id": run_id, "command": command, "config": resolved})
            continue
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
        if slurm.get("exclude"):
            exclude = slurm["exclude"]
            sbatch.append(f"--exclude={exclude if isinstance(exclude, str) else ','.join(exclude)}")
        # Exported into every job of the run; the shared wrappers act on them.
        exports = {
            "FL_BRATS_REQUIRE_GPU": "1" if slurm.get("require_gpu") else "0",
            "FL_BRATS_BIND_SOURCE": "1" if slurm.get("bind_source") else "0",
        }
        sbatch.append("--export=ALL," + ",".join(f"{k}={v}" for k, v in exports.items()))
        interconnect = resolved.get("communication_measurement", {}).get(
            "slurm_interconnect", {}
        )
        if interconnect.get("enabled"):
            sbatch.append(
                f"--acctg-freq=network={int(interconnect.get('sampling_seconds', 30))}"
            )
        if previous_job:
            # Apply the gate to both members of a federated run. The server
            # and clients must start concurrently once the predecessor ends.
            sbatch.append(f"--dependency=afterany:{previous_job}")
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
            # Clients exit after the last round while the server still runs the
            # best/final evaluation, so only cancel a server that outlives a grace
            # period (it is then stuck, e.g. waiting for a client that died).
            grace_minutes = int(slurm.get("finalize_grace_minutes", 180))
            cleanup_job = None if args.no_cleanup else subprocess.check_output(
                [
                    "sbatch", "--parsable", f"--partition={slurm['partition']}",
                    f"--dependency=afterany:{client_job}",
                    "--cpus-per-task=1", "--mem=256M", f"--time={grace_minutes + 10}",
                    f"--output={log_dir}/cleanup-%j.out", f"--error={log_dir}/cleanup-%j.err",
                    f"--wrap=for _ in $(seq {grace_minutes}); do "
                    f"squeue -h -j {server_job} 2>/dev/null | grep -q . || exit 0; sleep 60; "
                    f"done; scancel {server_job} || true",
                ],
                cwd=REPO_ROOT, text=True,
            ).strip()
            jobs = {"server": server_job, "clients": client_job}
            if cleanup_job:
                jobs["cleanup"] = cleanup_job
            missing = resolved.get("missing_modalities", {})
            evaluation = missing.get("evaluation", {}) if isinstance(missing, dict) else {}
            # Authoritative held-out best/final evaluation now runs on the server.
            # Retain the old client jobs only as explicitly requested diagnostics.
            evaluation = {**evaluation, **resolved.get("evaluation", {})}
            if bool(evaluation.get("client_diagnostics", False)) and bool(
                evaluation.get("evaluate_all_non_empty_combinations", True)
            ):
                evaluation_sbatch = [
                    item for item in sbatch if not item.startswith("--dependency=")
                ]

                aliases = []
                if bool(evaluation.get("evaluate_best_checkpoint", False)):
                    aliases.append("best")
                if bool(evaluation.get("evaluate_final_checkpoint", False)):
                    aliases.append("final")
                for alias in aliases:
                    evaluation_dir = run_dir / "evaluation" / alias
                    evaluation_job = subprocess.check_output(
                        evaluation_sbatch
                        + [
                            f"--dependency=afterany:{cleanup_job or server_job}",
                            f"--array=0-{clients - 1}",
                            f"--job-name=flb-eval-{alias}-{_slug(variant)[:30]}",
                            f"--output={log_dir}/evaluate-{alias}-%A-%a.out",
                            f"--error={log_dir}/evaluate-{alias}-%A-%a.err",
                            str(
                                REPO_ROOT
                                / "slurm"
                                / "_shared"
                                / "evaluate_federated_checkpoint.sbatch"
                            ),
                            str(runtime_config),
                            str(run_dir / "checkpoints" / f"{alias}.pt"),
                            str(evaluation_dir),
                            "client_",
                            use_apptainer,
                            image,
                        ],
                        cwd=REPO_ROOT,
                        text=True,
                    ).strip()
                    aggregate_job = subprocess.check_output(
                        [
                            "sbatch",
                            "--parsable",
                            f"--partition={slurm['partition']}",
                            f"--dependency=afterok:{evaluation_job}",
                            f"--output={log_dir}/aggregate-{alias}-%j.out",
                            f"--error={log_dir}/aggregate-{alias}-%j.err",
                            str(
                                REPO_ROOT / "slurm" / "_shared" / "run_experiment.sbatch"
                            ),
                            use_apptainer,
                            image,
                            "fedbrats",
                            "aggregate-federated-checkpoint-evaluation",
                            "--input-dir",
                            str(evaluation_dir),
                        ],
                        cwd=REPO_ROOT,
                        text=True,
                    ).strip()
                    jobs[f"{alias}_evaluation"] = evaluation_job
                    jobs[f"{alias}_evaluation_aggregate"] = aggregate_job
            (log_dir / "slurm-job-ids.json").write_text(
                json.dumps(jobs, indent=2) + "\n", encoding="utf-8"
            )
            submitted.append(
                {"run_id": run_id, "run_dir": str(run_dir), "log_dir": str(log_dir), "jobs": jobs}
            )
            previous_job = cleanup_job or server_job
            continue
        sbatch.extend(
            [
                f"--output={log_dir}/slurm-%j.out",
                f"--error={log_dir}/slurm-%j.err",
                # The federated jobs' wrapper: .env, dataset binds, GPU check, source bind.
                str(REPO_ROOT / "slurm" / "_shared" / "run_with_configured_environment.sh"),
                use_apptainer, image,
                *command,
            ]
        )
        job_id = subprocess.check_output(sbatch, cwd=REPO_ROOT, text=True).strip()
        (log_dir / "slurm-job-id.txt").write_text(job_id + "\n", encoding="utf-8")
        submitted.append(
            {
                "run_id": run_id, "run_dir": str(run_dir), "log_dir": str(log_dir),
                "job_id": job_id, "jobs": {"job": job_id},
            }
        )
        previous_job = job_id
    print(json.dumps(submitted, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
