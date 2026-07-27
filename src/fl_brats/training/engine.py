"""One training/evaluation engine for centralized and federated paths."""

from typing import Any

import torch
from torch.utils.data import DataLoader

from fl_brats.metrics.dice import dice_metrics


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def forward_batch(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    """Pass explicit masks when a model advertises mask support."""
    if getattr(model, "supports_modality_mask", False):
        return model(batch["image"], batch.get("modality_mask"))
    return model(batch["image"])


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total = 0.0
    samples = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(forward_batch(model, batch), batch["label"])
        loss.backward()
        optimizer.step()
        n = len(batch["image"])
        total += float(loss.detach()) * n
        samples += n
    return {"train_loss": total / samples if samples else 0.0, "samples": float(samples)}


def train_local_epochs(*args: Any, epochs: int = 1, **kwargs: Any) -> dict[str, float]:
    result: dict[str, float] = {}
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
        logits = forward_batch(model, batch)
        n = len(logits)
        metrics = dice_metrics(logits, batch["label"])
        aggregate["loss"] += float(loss_fn(logits, batch["label"])) * n
        for key, value in metrics.items():
            aggregate[key] += value * n
        samples += n
    return {key: value / samples if samples else 0.0 for key, value in aggregate.items()} | {
        "subjects": float(samples)
    }
