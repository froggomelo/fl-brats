"""One training/evaluation engine for centralized and federated paths."""

import multiprocessing
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import torch
from torch.utils.data import DataLoader

from fl_brats.metrics.segmentation import REGIONS, _hd95, segmentation_metrics
from fl_brats.models.fedamm import FedAMMOutput, fedamm_auxiliary_loss, semantic_prototypes


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def forward_batch(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    """Pass explicit masks when a model advertises mask support."""
    if getattr(model, "model_kind", "") == "sam" and batch["image"].ndim == 5:
        image = batch["image"]
        batch_size, channels, height, width, depth = image.shape
        slices = image.permute(0, 4, 1, 2, 3).reshape(
            batch_size * depth, channels, height, width
        )
        slice_batch_size = int(getattr(model, "config", {}).get("slice_batch_size", 8))
        logits = torch.cat(
            [
                model(slices[start : start + slice_batch_size])
                for start in range(0, len(slices), slice_batch_size)
            ]
        )
        return logits.reshape(batch_size, depth, 3, height, width).permute(0, 2, 3, 4, 1)
    if getattr(model, "supports_modality_mask", False):
        return model(batch["image"], batch.get("modality_mask"))
    return model(batch["image"])


def logits_from_output(output: torch.Tensor | FedAMMOutput) -> torch.Tensor:
    return output.logits if isinstance(output, FedAMMOutput) else output


def batch_loss(model_output: torch.Tensor | FedAMMOutput, target: torch.Tensor, loss_fn: Any, model: torch.nn.Module) -> torch.Tensor:
    loss = loss_fn(logits_from_output(model_output), target)
    if isinstance(model_output, FedAMMOutput) and (
        getattr(model, "prototype_distillation_enabled", True)
        or getattr(model, "global_prototype_alignment_enabled", True)
    ):
        weight = float(getattr(model, "prototype_loss_weight", 1.0))
        loss = loss + fedamm_auxiliary_loss(
            model_output,
            target,
            weight,
            distillation_enabled=bool(getattr(model, "prototype_distillation_enabled", True)),
        )
    return loss


def proximal_penalty(
    model: torch.nn.Module,
    global_parameters: list[torch.Tensor] | None,
    device: torch.device,
) -> torch.Tensor:
    """Return sum(||w_local - w_global||^2) over trainable parameters."""
    if global_parameters is None:
        return torch.zeros((), device=device)
    penalty = torch.zeros((), device=device)
    for local, global_parameter in zip(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        global_parameters,
        strict=True,
    ):
        penalty = penalty + torch.sum((local - global_parameter.to(device=device)) ** 2)
    return penalty


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    proximal_mu: float = 0.0,
    proximal_global_parameters: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    model.train()
    total = 0.0
    samples = 0
    prototype_sums: dict[int, torch.Tensor] = {}
    prototype_counts: dict[int, int] = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = forward_batch(model, batch)
        loss = batch_loss(output, batch["label"], loss_fn, model)
        if proximal_mu:
            loss = loss + 0.5 * proximal_mu * proximal_penalty(
                model, proximal_global_parameters, device
            )
        loss.backward()
        optimizer.step()
        n = len(batch["image"])
        total += float(loss.detach()) * n
        samples += n
        if isinstance(output, FedAMMOutput) and getattr(model, "emit_prototypes", True):
            prototypes = semantic_prototypes(output.fused_features.detach(), batch["label"].detach()).cpu()
            for combo_id, prototype in zip(output.combination_ids.detach().cpu().tolist(), prototypes, strict=True):
                key = int(combo_id)
                prototype_sums[key] = prototype if key not in prototype_sums else prototype_sums[key] + prototype
                prototype_counts[key] = prototype_counts.get(key, 0) + 1
    result: dict[str, Any] = {"train_loss": total / samples if samples else 0.0, "samples": float(samples)}
    if prototype_sums:
        import json

        result["fedamm_prototypes_json"] = json.dumps(
            {
                str(key): {
                    "count": prototype_counts[key],
                    "prototype": (value / prototype_counts[key]).tolist(),
                }
                for key, value in prototype_sums.items()
            },
            sort_keys=True,
        )
    return result


def train_local_epochs(*args: Any, epochs: int = 1, **kwargs: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for _ in range(epochs):
        result = train_one_epoch(*args, **kwargs)
    return result


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    loss_fn: Any,
    device: torch.device,
    *,
    metric_names: tuple[str, ...] = ("dice", "hd95", "sensitivity"),
    timing: dict[str, float] | None = None,
    hd95_workers: int = 0,
) -> dict[str, float]:
    """Average loss and segmentation metrics over the loader.

    HD95 runs a single-threaded CPU distance transform that takes seconds per
    subject for scattered predictions. With ``hd95_workers`` it runs in a
    process pool while the GPU moves on; results are summed in batch order, so
    the averages are identical to the serial computation.
    """
    model.eval()
    aggregate: dict[str, float] = {"loss": 0.0}
    for metric in metric_names:
        aggregate.update({f"{region}_{metric}": 0.0 for region in REGIONS})
        aggregate[f"mean_{metric}"] = 0.0
    samples = 0
    parallel_hd95 = "hd95" in metric_names and hd95_workers > 0
    inline = tuple(name for name in metric_names if name != "dice" and not (parallel_hd95 and name == "hd95"))
    pending: deque[tuple[Any, int]] = deque()
    batch_rows: list[dict[str, float]] = []
    started_expensive = time.perf_counter()
    pool = (
        ProcessPoolExecutor(max_workers=hd95_workers, mp_context=multiprocessing.get_context("spawn"))
        if parallel_hd95
        else None
    )
    try:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = logits_from_output(forward_batch(model, batch))
            n = len(logits)
            metrics = segmentation_metrics(logits, batch["label"], metric_names=("dice",)) if "dice" in metric_names else {}
            if inline:
                started = time.perf_counter()
                metrics.update(segmentation_metrics(logits, batch["label"], metric_names=inline))
                if timing is not None:
                    timing["expensive_metrics_seconds"] = timing.get("expensive_metrics_seconds", 0.0) + time.perf_counter() - started
            if pool is not None:
                prediction = (torch.sigmoid(logits) >= 0.5).cpu()
                pending.append((pool.submit(_hd95, prediction, batch["label"].bool().cpu()), len(batch_rows)))
                # Bound in-flight volumes (~27 MB each) so memory stays flat.
                while len(pending) > 2 * hd95_workers:
                    _store_hd95(batch_rows, *pending.popleft())
            metrics["loss"] = float(loss_fn(logits, batch["label"]))
            metrics["_n"] = float(n)
            batch_rows.append(metrics)
            samples += n
        while pending:
            _store_hd95(batch_rows, *pending.popleft())
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    if pool is not None and timing is not None:
        timing["expensive_metrics_seconds"] = timing.get("expensive_metrics_seconds", 0.0) + time.perf_counter() - started_expensive
    for metrics in batch_rows:
        n = metrics.pop("_n")
        for key, value in metrics.items():
            aggregate[key] += value * n
    return {key: value / samples if samples else 0.0 for key, value in aggregate.items()} | {
        "subjects": float(samples)
    }


def _store_hd95(rows: list[dict[str, float]], future: Any, index: int) -> None:
    values = future.result()
    for region, value in zip(REGIONS, values, strict=True):
        rows[index][f"{region}_hd95"] = value
    rows[index]["mean_hd95"] = sum(values) / len(REGIONS)
