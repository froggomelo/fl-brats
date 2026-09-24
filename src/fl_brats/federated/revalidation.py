"""Non-destructive checkpoint search and separate authoritative test reporting."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import yaml

from fl_brats.data.hashing import sha256_file, sha256_payload
from fl_brats.models.factory import create_model

from .evaluation import HeldOutEvaluator
from .research import atomic_write_json


def discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    rounds = []
    for path in (run_dir / "checkpoints").glob("round_*.pt"):
        match = re.fullmatch(r"round_(\d+)\.pt", path.name)
        if match:
            rounds.append((int(match[1]), path))
    rounds.sort(key=lambda item: item[0])
    if not rounds:
        raise ValueError(f"no round checkpoints in {run_dir}/checkpoints")
    if len({number for number, _ in rounds}) != len(rounds):
        raise ValueError("duplicate numeric round checkpoints")
    return rounds


def run_values(run_dir: Path, device: str | None = None) -> dict[str, Any]:
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        raise ValueError(f"missing saved preprocessing/loss configuration: {path}")
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if device is not None:
        values["device"] = device
    return values


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


@contextmanager
def evaluation_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"another evaluation owns {output}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def original_best_round(run_dir: Path) -> int | None:
    for name in ("summary.json", "state.json"):
        path = run_dir / name
        if path.exists():
            value = json.loads(path.read_text()).get("best_round")
            if value is not None:
                return int(value)
    path = run_dir / "checkpoints/best.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("round") is not None:
            return int(payload["round"])
    return None


def revalidate_run(
    run_dir: Path,
    split_path: Path,
    *,
    index_path: Path | None = None,
    device: str | None = None,
    expensive: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    checkpoints = discover_checkpoints(run_dir)
    values = run_values(run_dir, device)
    evaluator = HeldOutEvaluator(values, split_path, index_path=index_path)
    metadata = evaluator.metadata | {
        "original_run_path": str(run_dir),
        "historical_round_evaluation": "client training-partition evaluation",
        "corrected_revalidation": "fixed held-out validation evaluation",
        "expensive_metrics_requested": expensive,
    }
    identity = sha256_payload(metadata)
    output = run_dir / "revalidation"
    plan = {
        "original_run_path": str(run_dir),
        "total_checkpoints_discovered": len(checkpoints),
        "rounds": [r for r, _ in checkpoints],
        "output": str(output),
        "validation_subject_count": len(evaluator.subjects),
        "evaluation_identity": identity,
        "validation_subject_list_hash": metadata["validation_subject_list_hash"],
        "index_available": evaluator.index_path.is_file(),
        "dry_run": dry_run,
    }
    if dry_run:
        return plan
    evaluator.prepare()
    with evaluation_lock(output):
        metadata_path = output / "metadata.json"
        if metadata_path.exists():
            prior = json.loads(metadata_path.read_text())
            if prior["evaluation_identity"] != identity:
                raise ValueError("revalidation identity changed; use a separate run copy/output")
        else:
            if any(p.name != ".lock" for p in output.iterdir()):
                raise ValueError("unrecognized existing revalidation outputs")
            atomic_write_json(metadata_path, metadata | {"evaluation_identity": identity})
        rows_path = output / "validation_rounds.jsonl"
        rows = (
            [json.loads(line) for line in rows_path.read_text().splitlines()]
            if rows_path.exists()
            else []
        )
        completed = {row["round"]: row for row in rows}
        paths = dict(checkpoints)
        if len(completed) != len(rows) or not set(completed) <= set(paths):
            raise ValueError("saved revalidation rounds no longer match checkpoints")
        for number, row in completed.items():
            if sha256_file(paths[number]) != row["checkpoint_hash"]:
                raise ValueError(f"checkpoint changed since evaluation: {paths[number]}")
        atomic_write_json(output / "state.json", {"status": "running", "completed": len(rows)})
        for number, checkpoint in checkpoints:
            if number in completed:
                continue
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if int(payload["round"]) != number:
                raise ValueError(f"checkpoint round disagrees with filename: {checkpoint}")
            model = create_model(payload["model_config"])
            model.load_state_dict(payload["model_state"], strict=True)
            metrics, _ = evaluator.run(model, expensive=expensive)
            if not math.isfinite(metrics["mean_dice"]):
                raise ValueError(f"non-finite validation score for {checkpoint}")
            rows.append(
                {
                    "round": number,
                    **metrics,
                    "checkpoint": str(checkpoint),
                    "checkpoint_hash": sha256_file(checkpoint),
                }
            )
            rows.sort(key=lambda row: row["round"])
            atomic_jsonl(rows_path, rows)
            atomic_write_json(output / "state.json", {"status": "running", "completed": len(rows)})
            print(f"Revalidation round={number} mean_dice={metrics['mean_dice']:.6f}", flush=True)
            del model
        best = max(rows, key=lambda row: row["mean_dice"])
        original = original_best_round(run_dir)
        atomic_copy(paths[best["round"]], output / "best_revalidated.pt")
        summary = {
            **plan,
            "checkpoints_successfully_evaluated": len(rows),
            "best_revalidated_round": best["round"],
            "best_revalidated_mean_dice": best["mean_dice"],
            "original_best_round": original,
            "best_round_changed": best["round"] != original if original is not None else None,
            "historical_round_evaluation": metadata["historical_round_evaluation"],
            "corrected_revalidation": metadata["corrected_revalidation"],
            "selection_criterion": "held_out_natural_validation_mean_dice",
        }
        atomic_write_json(output / "summary.json", summary)
        atomic_write_json(output / "state.json", {"status": "completed", "completed": len(rows)})
        return summary


def evaluate_final_run(
    run_dir: Path,
    split_path: Path,
    *,
    checkpoint: Path | None = None,
    final_round: int | None = None,
    index_path: Path | None = None,
    device: str | None = None,
    robustness: bool | None = None,
    output_dir: Path | None = None,
) -> Path:
    if (checkpoint is None) == (final_round is None):
        raise ValueError("specify exactly one of checkpoint or explicit final_round")
    if final_round is not None:
        checkpoint = Path("checkpoints") / f"round_{final_round:04d}.pt"
    assert checkpoint is not None
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    destination = output_dir or run_dir / (
        "final_evaluation" if final_round is None else f"final_evaluation_round_{final_round:04d}"
    )
    if destination.exists():
        raise ValueError(f"refusing to overwrite final evaluation: {destination}")
    evaluator = HeldOutEvaluator(
        run_values(run_dir, device), split_path, cohort="test", index_path=index_path
    )
    if robustness is not None:
        evaluator.robustness_enabled = robustness
    evaluator.metadata.update(
        selection_criterion=None,
        checkpoint_role=("selected_best" if final_round is None else "predetermined_final"),
        explicit_final_round=final_round,
    )
    evaluator.evaluate_checkpoint(checkpoint, destination)
    return destination
