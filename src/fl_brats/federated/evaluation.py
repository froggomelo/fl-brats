"""Fixed research evaluation cohorts, never federated training data.

Validation selects checkpoints; test is accessible only to explicit final reporting.
Natural masks depend on the evaluation policy seed (default 42), never the run seed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from fl_brats.data.brats import (
    build_dataloader,
    build_dataset,
    persistent_cache_dir,
    records_from_index,
)
from fl_brats.data.cohorts import read_split, split_path_for
from fl_brats.data.hashing import sha256_file, sha256_payload
from fl_brats.data.modalities import combination_to_mask, policy_from_config
from fl_brats.data.transforms import build_transforms
from fl_brats.losses.segmentation import create_loss
from fl_brats.training.engine import evaluate
from fl_brats.training.evaluation import evaluate_modality_combinations

from .parameters import load_arrays, state_to_arrays
from .research import atomic_write_json


def evaluation_policy(values: dict[str, Any]) -> dict[str, Any]:
    # Retain the existing missing_modalities.evaluation keys; top-level settings win.
    policy = dict(values.get("missing_modalities", {}).get("evaluation", {}))
    policy.update(values.get("evaluation", {}))
    for key, default in (
        ("natural_every_rounds", 1),
        ("expensive_metrics_every_rounds", 10),
        ("robustness_every_rounds", 10),
    ):
        policy.setdefault(key, default)
        if int(policy[key]) < (1 if key == "natural_every_rounds" else 0):
            raise ValueError(f"invalid evaluation.{key}")
    policy.setdefault("evaluate_best_checkpoint", True)
    policy.setdefault("evaluate_final_checkpoint", True)
    policy.setdefault("client_diagnostics", False)
    return policy


def canonical_metrics(metrics: dict[str, float]) -> dict[str, float]:
    aliases = {
        f"{region}_{metric}": f"{metric}_{region}"
        for region in ("wt", "tc", "et")
        for metric in ("dice", "hd95", "sensitivity")
    }
    return {aliases.get(key, key): value for key, value in metrics.items()}


class HeldOutEvaluator:
    def __init__(
        self,
        values: dict[str, Any],
        split_path: Path | None = None,
        *,
        cohort: str = "validation",
        index_path: Path | None = None,
    ) -> None:
        if cohort not in {"validation", "test"}:
            raise ValueError("evaluation cohort must be validation or test")
        self.values, self.cohort = values, cohort
        self.policy = evaluation_policy(values)
        self.split_path = (split_path or split_path_for(values)).resolve()
        split = read_split(self.split_path)
        self.subjects = sorted(split[f"{cohort}_subjects"])
        if not self.subjects:
            raise ValueError(f"empty {cohort} cohort")
        self.index_path = index_path or Path(
            values.get("data", {}).get("index_path") or split["dataset_index_path"]
        )
        if not self.index_path.is_absolute() and not self.index_path.exists():
            self.index_path = self.split_path.parent / self.index_path
        self.modality_policy = dict(values.get("missing_modalities", {}))
        self.modality_policy.pop("evaluation", None)
        self.modality_policy.update(self.policy.get("natural_missingness", {}))
        self.modality_policy["seed"] = int(self.policy.get("modality_seed", 42))
        missingness = policy_from_config(self.modality_policy)
        self.masks = {
            subject: combination_to_mask(
                missingness.available_modalities(
                    subject_id=subject, client_id="global_evaluation", split=cohort
                )
            )
            for subject in self.subjects
        }
        if any(not mask.any() for mask in self.masks.values()):
            raise ValueError("evaluation policy produced an empty modality mask")
        self.loader: Any = None
        device = str(values.get("device", "auto"))
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        self.loss_fn = create_loss(values.get("loss", {}))
        self.robustness_enabled = bool(
            values.get("missing_modalities", {}).get("enabled", False)
        ) and bool(self.policy.get("evaluate_all_non_empty_combinations", True))
        self.metadata = {
            "cohort": cohort,
            "split_path": str(self.split_path),
            "split_file_hash": sha256_file(self.split_path),
            "validation_subject_count": len(split["validation_subjects"]),
            "validation_subject_list_hash": sha256_payload(sorted(split["validation_subjects"])),
            "test_subject_list_hash": sha256_payload(sorted(split["test_subjects"])),
            "subject_count": len(self.subjects),
            "subject_ids": self.subjects,
            "index_path": str(self.index_path),
            "expected_index_hash": split.get("dataset_index_hash"),
            "preprocessing_config": values.get("data", {}),
            "preprocessing_config_hash": sha256_payload(values.get("data", {})),
            "loss_config": values.get("loss", {}),
            "modality_policy": self.modality_policy,
            "natural_masks": {key: mask.tolist() for key, mask in self.masks.items()},
            "metric_policy": {
                "threshold": 0.5,
                "aggregation": "subject_macro",
                "hd95_units": "preprocessed_voxels",
                "empty_masks": "both empty: Dice/sensitivity=1, HD95=0; "
                "one empty: HD95=spatial diagonal; empty truth sensitivity=0",
            },
            "cadence": self.policy,
            "selection_criterion": "held_out_natural_validation_mean_dice"
            if cohort == "validation"
            else None,
            "data_roles": {
                "clients": "training only",
                "validation": "model selection",
                "test": "final reporting only",
            },
        }

    def prepare(self) -> None:
        if self.loader is not None:
            return
        expected = self.metadata["expected_index_hash"]
        if expected and sha256_file(self.index_path) != expected:
            raise ValueError("evaluation index differs from frozen split")
        records = records_from_index(self.index_path, self.subjects)
        data = self.values.get("data", {})
        cache_mode = str(data.get("cache_mode", "dataset"))
        self.loader = build_dataloader(
            build_dataset(
                records,
                build_transforms(data, False),
                cache_mode,
                persistent_cache_dir(data) if cache_mode == "persistent" else None,
            ),
            int(self.policy.get("batch_size", 1)),
            False,
            int(self.values.get("num_workers") or 0),
            42,
            self.values.get("pin_memory"),
        )

    def natural_batches(self):
        self.prepare()
        for original in self.loader:
            batch = dict(original)
            masks = torch.stack([self.masks[s] for s in batch["subject_id"]])
            image = batch["image"]
            batch["image"] = image * masks.to(image.device).view(
                len(masks), 4, *([1] * (image.ndim - 2))
            )
            batch["modality_mask"] = masks
            yield batch

    def run(
        self, model: torch.nn.Module, *, expensive: bool = False, robustness: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.prepare()
        model.to(self.device)
        started = time.perf_counter()
        timing: dict[str, float] = {}
        names = ("dice", "hd95", "sensitivity") if expensive else ("dice",)
        metrics = canonical_metrics(
            evaluate(
                model,
                self.natural_batches(),
                self.loss_fn,
                self.device,
                metric_names=names,
                timing=timing,
            )
        )
        natural_seconds = time.perf_counter() - started
        combinations = {}
        robustness_started = time.perf_counter()
        if robustness:
            combinations = {
                key: canonical_metrics(value)
                for key, value in evaluate_modality_combinations(
                    model, self.loader, self.loss_fn, self.device, metric_names=names
                ).items()
            }
        metrics.update(
            evaluation_seconds=time.perf_counter() - started,
            natural_evaluation_seconds=natural_seconds,
            expensive_metrics_seconds=timing.get("expensive_metrics_seconds", 0.0),
            robustness_seconds=time.perf_counter() - robustness_started if robustness else 0.0,
            expensive_metrics_computed=expensive,
            robustness_computed=robustness,
        )
        metrics[f"natural_{self.cohort}_seconds"] = natural_seconds
        return metrics, combinations

    def callback(self, model: torch.nn.Module, run_dir: Path):
        if self.cohort != "validation":
            raise ValueError("test evaluation cannot select checkpoints")
        atomic_write_json(run_dir / "evaluation_metadata.json", self.metadata)
        print(
            f"Global validation: {len(self.subjects)} held-out subjects; "
            f"hash={self.metadata['validation_subject_list_hash']}",
            flush=True,
        )

        def evaluate_round(round_number: int, arrays: Any, config: Any):
            if round_number % int(self.policy["natural_every_rounds"]):
                return None
            load_arrays(model, state_to_arrays(model)[0], arrays)
            expensive_interval = int(self.policy["expensive_metrics_every_rounds"])
            robustness_interval = int(self.policy["robustness_every_rounds"])
            metrics, combinations = self.run(
                model,
                expensive=bool(expensive_interval and round_number % expensive_interval == 0),
                robustness=self.robustness_enabled
                and bool(robustness_interval and round_number % robustness_interval == 0),
            )
            if combinations:
                from .research import RoundArtifactWriter

                RoundArtifactWriter(run_dir)._append(
                    run_dir / "metrics/modality_combinations.jsonl",
                    {"round": round_number, "cohort": "validation", "combinations": combinations},
                )
            print(
                f"Global validation round={round_number} subjects={metrics['subjects']} "
                f"mean_dice={metrics['mean_dice']:.6f}",
                flush=True,
            )
            return metrics["loss"], metrics

        return evaluate_round

    def evaluate_checkpoint(self, checkpoint: Path, destination: Path) -> None:
        from fl_brats.models.factory import create_model

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        explicit_round = self.metadata.get("explicit_final_round")
        if explicit_round is not None and int(payload["round"]) != explicit_round:
            raise ValueError("checkpoint does not match the requested predetermined final round")
        model = create_model(payload["model_config"])
        model.load_state_dict(payload["model_state"], strict=True)
        metrics, combinations = self.run(model, expensive=True, robustness=self.robustness_enabled)
        destination.mkdir(parents=True, exist_ok=False)
        atomic_write_json(destination / "metrics.json", metrics)
        atomic_write_json(
            destination / "metadata.json",
            self.metadata
            | {
                "checkpoint": str(checkpoint),
                "checkpoint_hash": sha256_file(checkpoint),
                "checkpoint_round": payload["round"],
            },
        )
        (destination / "modality_combinations.jsonl").write_text(
            "".join(
                json.dumps({"combination": key, **value}) + "\n"
                for key, value in combinations.items()
            ),
            encoding="utf-8",
        )
