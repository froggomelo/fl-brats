#!/usr/bin/env python3
"""Keep a prioritized set of federated runs moving on a flaky Slurm cluster.

Run ``tick`` every few minutes from cron on the login node. Each tick
reconciles submitted attempts with ``squeue`` and the run directories,
cancels attempts where a member died or no round finished for too long,
excludes nodes whose GPU driver is broken, and resubmits unfinished runs
from their latest round checkpoint, highest priority first, within the
per-partition limits of the campaign file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import getpass
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from launch_experiment import _load_experiment_yaml  # noqa: E402
GPU_MARKER = re.compile(r"FL_BRATS_GPU_UNAVAILABLE host=(\S+)")
DONE = {"completed", "early_stopped"}
# Pending jobs that do not compete for the per-user running-job limit yet.
NOT_WAITING_FOR_A_SLOT = {"Dependency", "DependencyNeverSatisfied", "JobHeldUser", "JobHeldAdmin"}


def _now() -> float:
    return time.time()


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Campaign:
    def __init__(self, spec_path: Path) -> None:
        self.spec_path = spec_path.resolve()
        self.spec = yaml.safe_load(self.spec_path.read_text(encoding="utf-8"))
        # Monitoring logs live under logs/monitoring/ (Experiments.md layout).
        self.root = REPO_ROOT / "logs" / "monitoring" / "campaign" / str(self.spec["name"])
        legacy = REPO_ROOT / "logs" / "campaign" / str(self.spec["name"])
        if legacy.is_dir() and not self.root.exists():
            self.root.parent.mkdir(parents=True, exist_ok=True)
            legacy.rename(self.root)  # one-time move of a campaign started before the layout change
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state: dict[str, Any] = (
            json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state_path.exists()
            else {"units": {}, "bad_nodes": {}}
        )
        for node, reason in (self.spec.get("known_bad_nodes") or {}).items():
            self.state["bad_nodes"].setdefault(node, reason)
        for unit in self.units():
            record = self.state["units"].setdefault(
                unit["key"], {"status": "pending", "attempts": [], "no_progress": 0}
            )
            if unit["existing"] and not record["attempts"]:
                # Adopt a run from before the campaign so it resumes rather than restarts.
                record["attempts"].append(
                    {
                        "run_id": Path(unit["existing"]).name, "run_dir": unit["existing"],
                        "log_dir": "", "jobs": {}, "partition": "-", "start_round": 0,
                        "ended": _now(), "end_reason": "adopted pre-campaign run",
                    }
                )

    # ----------------------------------------------------------------- helpers
    def units(self) -> list[dict[str, Any]]:
        units = []
        for entry in self.spec["runs"]:
            for seed in entry.get("seeds", [42]):
                units.append(
                    {
                        "key": f"{entry['config']}#seed{seed}",
                        "config": entry["config"],
                        "seed": int(seed),
                        "partitions": list(entry.get("partitions", ["batch"])),
                        "set": list(entry.get("set", [])),
                        "existing": (entry.get("existing") or {}).get(seed),
                    }
                )
        return units

    def log(self, message: str) -> None:
        line = f"{_stamp()} {message}"
        print(line)
        with (self.root / "events.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def save(self) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)

    @staticmethod
    def queue() -> dict[str, list[tuple[str, str]]]:
        """Map each job or array master ID to its (state, node) entries."""
        output = subprocess.run(
            ["squeue", "-h", "-r", "-u", getpass.getuser(), "-o", "%F|%T|%N"],
            check=True, capture_output=True, text=True,
        ).stdout
        jobs: dict[str, list[tuple[str, str]]] = {}
        for line in output.splitlines():
            master, state, node = (line.split("|") + ["", ""])[:3]
            jobs.setdefault(master.strip(), []).append((state.strip(), node.strip()))
        return jobs

    @staticmethod
    def jobs_holding_slots(partition: str) -> int:
        """Our jobs on a partition that run or will run as soon as a slot frees."""
        output = subprocess.run(
            ["squeue", "-h", "-r", "-u", getpass.getuser(), "-p", partition, "-o", "%T|%r"],
            check=True, capture_output=True, text=True,
        ).stdout
        count = 0
        for line in output.splitlines():
            state, reason = (line.split("|") + [""])[:2]
            # Held jobs appear as "JobHeldUser" or "job requeued in held state".
            held = "held" in reason.lower()
            if state == "RUNNING" or (state == "PENDING" and reason not in NOT_WAITING_FOR_A_SLOT and not held):
                count += 1
        return count

    def held_jobs(self, partition: str) -> dict[str, int]:
        """Jobs each active run on a partition still holds, from the queue
        (a run whose clients exited while its server finalizes holds one)."""
        queue = self.queue()
        held: dict[str, int] = {}
        for key, unit in self.state["units"].items():
            if unit["status"] != "active" or unit["attempts"][-1]["partition"] != partition:
                continue
            jobs = unit["attempts"][-1]["jobs"]
            held[key] = sum(len(queue.get(jobs[k], [])) for k in ("server", "clients", "job") if k in jobs)
        return held

    def training_jobs(self, partition: str) -> int:
        """Jobs the campaign's active training runs still hold on a partition."""
        return sum(self.held_jobs(partition).values())

    def fits(self, partition: str, needed: int, *, training: bool = True) -> bool:
        """Whether a run's jobs can all hold slots at once.

        A run whose jobs cannot all start deadlocks: the server waits for a
        client that waits for a slot. Every job of ours counts against the
        per-user limit (job_limit); training runs also stay within their own
        budget (max_jobs), so evaluation jobs never shrink the training lanes.
        """
        settings = self.spec["partitions"].get(partition, {})
        limit = settings.get("job_limit")
        if limit is not None and self.jobs_holding_slots(partition) + needed > int(limit):
            return False
        budget = settings.get("max_jobs") if training else None
        return budget is None or self.training_jobs(partition) + needed <= int(budget)

    def can_backfill(self, partitions: list[str], needed: int, blocked: int) -> bool:
        """Whether a later run may start while an earlier one waits.

        Only if the waiting run would still fit as soon as any training run
        moves to finalization (clients gone, the server alone holding one
        slot): budget minus what would then be held, minus this run's jobs,
        must cover it. In practice one centralized run can use the slot two
        five-job runs leave, and a five-job run never overtakes k8.
        """
        for partition in partitions:
            budget = int(self.spec["partitions"].get(partition, {}).get("max_jobs", 0))
            held = self.held_jobs(partition)
            total = sum(held.values())
            training = [count for count in held.values() if count > 1]
            after = [total - count + 1 for count in training] or [total]
            if all(budget - value - needed >= blocked for value in after):
                return True
        return False

    @staticmethod
    def jobs_needed(config: str) -> int:
        """Jobs a run holds at once: the server and one per client, or one centralized job."""
        values = _load_experiment_yaml(REPO_ROOT / config)
        if values["experiment"]["family"] == "centralized":
            return 1
        return 1 + int(values.get("federated", {}).get("num_clients", 4))

    def resume_point(self, config: str, attempts: list[dict[str, Any]]) -> tuple[int, Path | None]:
        """Where a retry continues: the latest round checkpoint, or for a
        centralized run the last.pt of the attempt with the most epochs."""
        if self.jobs_needed(config) > 1:
            return self.latest_checkpoint(attempts)
        best: tuple[int, Path | None] = (0, None)
        for attempt in attempts:
            epochs, _ = self.epochs_done(attempt)
            last = Path(attempt["run_dir"]) / "checkpoints" / "last.pt"
            if epochs > best[0] and last.is_file():
                best = (epochs, last)
        return best

    @staticmethod
    def epochs_done(attempt: dict[str, Any]) -> tuple[int, float]:
        """Completed epochs of a centralized attempt and when the last one finished."""
        path = Path(attempt["run_dir"]) / "metrics.jsonl"
        try:
            return len(path.read_text(encoding="utf-8").splitlines()), path.stat().st_mtime
        except OSError:
            return 0, 0.0

    @staticmethod
    def latest_checkpoint(attempts: list[dict[str, Any]]) -> tuple[int, Path | None]:
        best: tuple[int, Path | None] = (0, None)
        for attempt in attempts:
            for path in Path(attempt["run_dir"]).glob("checkpoints/round_*.pt"):
                number = int(path.stem.split("_")[1])
                if number > best[0]:
                    best = (number, path)
        return best

    @staticmethod
    def run_status(attempt: dict[str, Any]) -> str | None:
        path = Path(attempt["run_dir"]) / "state.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError):
            return None

    def cancel(self, attempt: dict[str, Any], reason: str) -> None:
        ids = [attempt["jobs"][k] for k in ("server", "clients", "cleanup", "job") if k in attempt["jobs"]]
        subprocess.run(["scancel", *ids], check=False)
        attempt["ended"] = _now()
        attempt["end_reason"] = reason

    def record_gpu_failures(self, attempt: dict[str, Any]) -> int:
        """Exclude nodes whose GPU check failed; return how many were new."""
        found = 0
        for path in Path(attempt["log_dir"]).glob("*.err"):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for node in GPU_MARKER.findall(text):
                if node not in self.state["bad_nodes"]:
                    self.state["bad_nodes"][node] = f"GPU driver unusable ({_stamp()})"
                    self.log(f"excluding {node}: GPU driver unusable")
                    found += 1
        return found

    # --------------------------------------------------------------- reconcile
    def sweep_orphans(self, queue: dict[str, list[tuple[str, str]]]) -> None:
        """Cancel jobs of ended attempts that are still queued.

        A cancel can fail silently; left alone, such orphans hold job slots
        (the per-user QOS limit) and starve the attempts that replaced them.
        """
        for key, unit in self.state["units"].items():
            for attempt in unit["attempts"]:
                if "ended" not in attempt:
                    continue
                alive = [job for job in attempt["jobs"].values() if job in queue]
                if alive:
                    subprocess.run(["scancel", *alive], check=False)
                    self.log(f"cancelled orphaned jobs {alive} of {key} ({attempt['run_id']})")

    def reconcile(self, queue: dict[str, list[tuple[str, str]]]) -> None:
        self.sweep_orphans(queue)
        stall_seconds = float(self.spec.get("stall_minutes", 75)) * 60
        for key, unit in self.state["units"].items():
            if unit["status"] != "active":
                continue
            attempt = unit["attempts"][-1]
            jobs = attempt["jobs"]
            if "job" in jobs:
                self.reconcile_single(key, unit, attempt, queue.get(jobs["job"], []), stall_seconds)
                continue
            server = queue.get(jobs["server"], [])
            clients = queue.get(jobs["clients"], [])
            expected = int(attempt.get("clients", 4))
            if self.record_gpu_failures(attempt):
                attempt["hit_bad_node"] = True
            running = bool(server) and server[0][0] == "RUNNING" and len(clients) == expected and all(
                state == "RUNNING" for state, _ in clients
            )
            if running and "running_since" not in attempt:
                attempt["running_since"] = _now()
                attempt["nodes"] = [node for _, node in server + clients]
            reason = None
            if not server and not clients:
                reason = "jobs ended"
            elif server and server[0][0] == "RUNNING" and 0 < len(clients) < expected:
                # After the last round every client is told to disconnect, and
                # one on a slow node can outlive the others by a tick; only a
                # client gone for 10 minutes while the rest wait is a crash.
                attempt.setdefault("clients_partial_since", _now())
                if _now() - attempt["clients_partial_since"] > 600:
                    reason = "a client ended while the others were still waiting"
            elif server and server[0][0] == "RUNNING" and not clients:
                # Clients exit after the last round while the server finalizes
                # (best/final evaluation). Finalization writes final.pt first, so
                # clients gone without final.pt means they crashed mid-training.
                attempt.setdefault("clients_gone_since", _now())
                gone = _now() - attempt["clients_gone_since"]
                finalizing = (Path(attempt["run_dir"]) / "checkpoints" / "final.pt").is_file()
                grace = float(self.spec.get("finalize_minutes", 180)) * 60
                if not finalizing and gone > 600:
                    reason = "all clients ended before training finished"
                elif gone > grace:
                    reason = "server still running long after all clients ended"
            elif not server and clients:
                reason = "server ended while clients remained"
            elif running:
                round_number, path = self.latest_checkpoint([attempt])
                last = max(attempt["running_since"], path.stat().st_mtime if path else 0.0)
                if _now() - last > stall_seconds:
                    reason = f"no round finished for {int((_now() - last) / 60)} min (stall)"
            if reason is None:
                continue
            if reason != "jobs ended":
                self.cancel(attempt, reason)
            else:
                attempt["ended"], attempt["end_reason"] = _now(), reason
                cleanup = jobs.get("cleanup")
                if cleanup and cleanup in queue:
                    subprocess.run(["scancel", cleanup], check=False)
            status = self.run_status(attempt)
            before = attempt.get("start_round", 0)
            after, _ = self.latest_checkpoint(unit["attempts"])
            if status in DONE:
                unit["status"], unit["final_run_dir"] = "done", attempt["run_dir"]
                self.log(f"DONE {key} at round {after} ({status}) -> {attempt['run_dir']}")
                continue
            if after > before:
                unit["no_progress"] = 0
            elif not attempt.get("hit_bad_node"):
                # A newly found dead-GPU node is the cluster's fault, not the run's.
                unit["no_progress"] = unit.get("no_progress", 0) + 1
            limit = int(self.spec.get("max_attempts_without_progress", 3))
            if unit["no_progress"] >= limit:
                unit["status"] = "stuck"
                self.log(f"STUCK {key}: {limit} attempts without progress; last: {reason}")
            else:
                unit["status"] = "pending"
                self.log(f"retry {key}: {reason}; resume from round {after}")

    def reconcile_single(
        self, key: str, unit: dict[str, Any], attempt: dict[str, Any],
        job: list[tuple[str, str]], stall_seconds: float,
    ) -> None:
        """A centralized run is one job; a retry resumes from its last completed epoch."""
        if self.record_gpu_failures(attempt):
            attempt["hit_bad_node"] = True
        if job and job[0][0] == "RUNNING" and "running_since" not in attempt:
            attempt["running_since"], attempt["nodes"] = _now(), [job[0][1]]
        epochs, last = self.epochs_done(attempt)
        reason = None
        if not job:
            reason = "jobs ended"
        elif "running_since" in attempt and _now() - max(attempt["running_since"], last) > stall_seconds:
            reason = f"no epoch finished for {int((_now() - max(attempt['running_since'], last)) / 60)} min (stall)"
        if reason is None:
            return
        if job:
            self.cancel(attempt, reason)
        else:
            attempt["ended"], attempt["end_reason"] = _now(), reason
        if (Path(attempt["run_dir"]) / "summary.json").is_file():
            unit["status"], unit["final_run_dir"] = "done", attempt["run_dir"]
            self.log(f"DONE {key} after {epochs} epochs -> {attempt['run_dir']}")
            return
        if epochs > 0:
            unit["no_progress"] = 0
        elif not attempt.get("hit_bad_node"):
            unit["no_progress"] = unit.get("no_progress", 0) + 1
        limit = int(self.spec.get("max_attempts_without_progress", 3))
        if unit["no_progress"] >= limit:
            unit["status"] = "stuck"
            self.log(f"STUCK {key}: {limit} attempts without progress; last: {reason}")
        else:
            unit["status"] = "pending"
            self.log(f"retry {key}: {reason} after {epochs} epochs")

    def archive_off_partition(self) -> None:
        """Set aside finished attempts made on a partition the unit may no longer use.

        Every final result must come from one GPU model, so a unit is never
        resumed from, evaluated from, or counted done by an attempt on another
        partition. Active attempts are left to finish; they are archived on the
        tick after they end and the unit then reruns from scratch.
        """
        for unit_spec in self.units():
            unit = self.state["units"][unit_spec["key"]]
            if unit["status"] == "active":
                continue
            allowed = set(unit_spec["partitions"]) | {"-"}
            moved = [a for a in unit["attempts"] if a["partition"] not in allowed]
            if not moved:
                continue
            unit.setdefault("archived_attempts", []).extend(moved)
            unit["attempts"] = [a for a in unit["attempts"] if a["partition"] in allowed]
            if unit["status"] in ("done", "stuck"):
                unit["status"], unit["no_progress"] = "pending", 0
                unit.pop("final_run_dir", None)
                unit.pop("posthoc", None)
            self.log(
                f"archived {len(moved)} attempt(s) of {unit_spec['key']} on "
                f"{', '.join(sorted({a['partition'] for a in moved}))}; rerun on {'/'.join(unit_spec['partitions'])}"
            )

    # ------------------------------------------------------------------ submit
    def submit(self) -> None:
        limits = {name: int(cfg.get("max_active", 1)) for name, cfg in self.spec["partitions"].items()}
        active: dict[str, int] = {name: 0 for name in limits}
        for unit in self.state["units"].values():
            if unit["status"] == "active":
                active[unit["attempts"][-1]["partition"]] = active.get(unit["attempts"][-1]["partition"], 0) + 1
        blocked: int | None = None  # jobs the first run that did not fit needs
        for unit_spec in self.units():
            unit = self.state["units"][unit_spec["key"]]
            if unit["status"] != "pending":
                continue
            needed = self.jobs_needed(unit_spec["config"])
            if blocked is not None and not self.can_backfill(unit_spec["partitions"], needed, blocked):
                continue
            partition = next(
                (
                    p for p in unit_spec["partitions"]
                    if active.get(p, 0) < limits.get(p, 0) and self.fits(p, needed)
                ),
                None,
            )
            if partition is None:
                # Priority: later runs may only use slots the blocked run could
                # never use (see can_backfill).
                if blocked is None:
                    blocked = needed
                continue
            round_number, checkpoint = self.resume_point(unit_spec["config"], unit["attempts"])
            command = [
                sys.executable, str(REPO_ROOT / "scripts" / "launch_experiment.py"),
                unit_spec["config"], "--mode", "slurm", "--seed", str(unit_spec["seed"]),
                "--set", f"slurm.partition={partition}",
                # The tick cancels a server stuck in finalization (finalize_minutes).
                "--no-cleanup",
            ]
            for assignment in list(self.spec["partitions"][partition].get("set", [])) + unit_spec["set"]:
                command += ["--set", assignment]
            if self.state["bad_nodes"]:
                command += ["--exclude", ",".join(sorted(self.state["bad_nodes"]))]
            if checkpoint is not None:
                command += ["--resume-checkpoint", str(checkpoint)]
            result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
            if result.returncode:
                self.log(f"submit failed for {unit_spec['key']}: {result.stderr.strip()[-400:]}")
                return  # most often a QOS submit limit; retry next tick
            submitted = json.loads(result.stdout)[0]
            values = yaml.safe_load(
                (Path(submitted["log_dir"]) / "resolved-launch-config.yaml").read_text(encoding="utf-8")
            )
            unit["attempts"].append(
                {
                    "run_id": submitted["run_id"], "run_dir": submitted["run_dir"],
                    "log_dir": submitted["log_dir"], "jobs": submitted["jobs"],
                    "partition": partition, "submitted": _now(), "start_round": round_number,
                    "clients": int(values.get("federated", {}).get("num_clients", 4)),
                }
            )
            unit["status"] = "active"
            active[partition] = active.get(partition, 0) + 1
            self.log(
                f"submitted {unit_spec['key']} on {partition} "
                f"({'resume from ' + str(round_number) if checkpoint else 'fresh'}) "
                f"job={submitted['jobs'].get('server', submitted['jobs'].get('job'))}"
            )

    # ---------------------------------------------------------------- post-hoc
    def posthoc_dirs(self, unit: dict[str, Any]) -> list[str]:
        """Run directories of the resume chain that hold round checkpoints, oldest first."""
        return [
            attempt["run_dir"]
            for attempt in unit["attempts"]
            if any(Path(attempt["run_dir"]).glob("checkpoints/round_*.pt"))
        ]

    def reconcile_posthoc(self, queue: dict[str, list[tuple[str, str]]]) -> None:
        """Track the periodic-evaluation job that follows each finished run."""
        for key, unit in self.state["units"].items():
            job = unit.get("posthoc")
            if not job or job["status"] != "active" or job["job"] in queue:
                continue
            state_path = Path(job["output"]) / "state.json"
            test_report = Path(job["output"]).parent / "final_evaluation" / "metrics.json"
            try:
                finished = (
                    json.loads(state_path.read_text()).get("status") == "completed"
                    and test_report.is_file()
                )
            except (OSError, ValueError):
                finished = False
            if finished:
                job["status"] = "done"
                self.log(f"DONE periodic evaluation {key}")
                continue
            self.record_gpu_failures({"log_dir": job["log_dir"]})
            job["attempts"] = job.get("attempts", 1)
            if job["attempts"] >= int(self.spec.get("max_attempts_without_progress", 3)) + 2:
                job["status"] = "stuck"
                self.log(f"STUCK periodic evaluation {key}")
            else:
                job["status"] = "pending"
                self.log(f"retry periodic evaluation {key}")

    def submit_posthoc(self) -> None:
        settings = self.spec.get("posthoc") or {}
        if not settings:
            return
        limit = int(settings.get("max_active", 1))
        active = sum(
            1 for unit in self.state["units"].values()
            if unit.get("posthoc", {}).get("status") == "active"
        )
        partition = settings.get("partition", "batch")
        for unit_spec in self.units():
            if active >= limit or not self.fits(partition, 1, training=False):
                return
            unit = self.state["units"][unit_spec["key"]]
            job = unit.get("posthoc")
            if unit["status"] != "done" or (job and job["status"] in ("active", "done", "stuck")):
                continue
            dirs = self.posthoc_dirs(unit)
            if not dirs:
                continue
            log_dir = Path(unit["attempts"][-1]["log_dir"] or self.root) / "periodic"
            log_dir.mkdir(parents=True, exist_ok=True)
            command = [
                "sbatch", "--parsable", f"--partition={partition}",
                f"--job-name=flb-periodic",
                f"--cpus-per-task={settings.get('cpus_per_task', 6)}",
                f"--mem={settings.get('memory', '29000M')}",
                f"--time={settings.get('time', '24:00:00')}",
                "--export=ALL,FL_BRATS_REQUIRE_GPU=1,FL_BRATS_BIND_SOURCE=1",
                f"--output={log_dir}/periodic-%j.out", f"--error={log_dir}/periodic-%j.err",
            ]
            if self.state["bad_nodes"]:
                command.append("--exclude=" + ",".join(sorted(self.state["bad_nodes"])))
            # Evaluate in the run's own image: SAM needs the image with its packages.
            run_values = _load_experiment_yaml(REPO_ROOT / unit_spec["config"])
            image = (run_values.get("slurm") or {}).get("image") or settings.get(
                "image", "apptainer/fl-brats-batch.sif"
            )
            command += [
                str(REPO_ROOT / "slurm" / "_shared" / "run_with_configured_environment.sh"),
                "true", str(image),
                "fedbrats", "evaluate-periodic-checkpoints",
                "--every", str(settings.get("every", 10)), "--device", "cuda", "--test-best",
            ]
            for run_dir in dirs:
                command += ["--run-dir", run_dir]
            result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
            if result.returncode:
                self.log(f"periodic submit failed for {unit_spec['key']}: {result.stderr.strip()[-300:]}")
                return
            attempts = (job or {}).get("attempts", 0) + 1
            unit["posthoc"] = {
                "status": "active", "job": result.stdout.strip(), "log_dir": str(log_dir),
                "output": str(Path(dirs[-1]) / "periodic_evaluation"), "attempts": attempts,
            }
            active += 1
            self.log(f"submitted periodic evaluation {unit_spec['key']} job={result.stdout.strip()}")

    # ------------------------------------------------------------------ status
    def status(self) -> str:
        rows = [f"{'run':70s} {'status':8s} {'round':>5s} {'idle':>6s} {'tries':>5s}  partition"]
        for unit_spec in self.units():
            unit = self.state["units"][unit_spec["key"]]
            round_number, path = self.latest_checkpoint(unit["attempts"])
            if unit["attempts"] and "job" in unit["attempts"][-1]["jobs"]:
                round_number, mtime = self.epochs_done(unit["attempts"][-1])
                path = Path(unit["attempts"][-1]["run_dir"]) / "metrics.jsonl" if mtime else None
            partition = unit["attempts"][-1]["partition"] if unit["attempts"] else "-"
            posthoc = unit.get("posthoc", {}).get("status", "-")
            # Minutes since the last round checkpoint: the stall signal for active runs.
            idle = f"{(_now() - path.stat().st_mtime) / 60:5.0f}m" if path and unit["status"] == "active" else "     -"
            rows.append(
                f"{unit_spec['key']:70s} {unit['status']:8s} {round_number:5d} {idle} "
                f"{len(unit['attempts']):5d}  {partition:9s} periodic={posthoc}"
            )
        rows.append("excluded nodes: " + ", ".join(sorted(self.state["bad_nodes"])))
        return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("tick", "status"))
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    campaign = Campaign(args.campaign)
    if args.command == "status":
        print(campaign.status())
        return 0
    with (campaign.root / "tick.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0  # a previous tick is still running
        queue = campaign.queue()
        campaign.reconcile(queue)
        campaign.reconcile_posthoc(queue)
        campaign.archive_off_partition()
        campaign.submit()
        campaign.submit_posthoc()
        campaign.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
