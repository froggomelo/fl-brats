"""One training/evaluation engine for centralized and federated paths."""

from typing import Any

import torch
from torch.utils.data import DataLoader

from fl_brats.metrics.dice import dice_metrics
from fl_brats.models.fedamm import FedAMMOutput, fedamm_auxiliary_loss, semantic_prototypes


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def forward_batch(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    """Pass explicit masks when a model advertises mask support."""
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


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
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
    model: torch.nn.Module, loader: DataLoader[Any], loss_fn: Any, device: torch.device
) -> dict[str, float]:
    model.eval()
    aggregate: dict[str, float] = {
        "loss": 0.0,
        "wt_dice": 0.0,
        "tc_dice": 0.0,
        "et_dice": 0.0,
        "mean_dice": 0.0,
    }
    samples = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = logits_from_output(forward_batch(model, batch))
        n = len(logits)
        metrics = dice_metrics(logits, batch["label"])
        aggregate["loss"] += float(loss_fn(logits, batch["label"])) * n
        for key, value in metrics.items():
            aggregate[key] += value * n
        samples += n
    return {key: value / samples if samples else 0.0 for key, value in aggregate.items()} | {
        "subjects": float(samples)
    }
