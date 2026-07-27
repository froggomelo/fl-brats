"""Controlled centralized epoch loop used by CLI orchestration."""

from typing import Any

import torch

from fl_brats.training.engine import evaluate, train_one_epoch


def train(
    model: torch.nn.Module,
    train_loader: Any,
    validation_loader: Any,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
) -> list[dict[str, float]]:
    history = []
    for epoch in range(1, epochs + 1):
        history.append(
            {"epoch": float(epoch)}
            | train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            | {
                f"validation_{k}": v
                for k, v in evaluate(model, validation_loader, loss_fn, device).items()
            }
        )
    return history
