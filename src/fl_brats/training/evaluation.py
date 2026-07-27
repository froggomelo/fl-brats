"""Evaluation helpers for naturally assigned and counterfactual modality masks."""

from typing import Any

import torch

from fl_brats.data.modalities import (
    all_non_empty_modality_combinations,
    combination_to_mask,
    modality_combination_key,
)

from .engine import evaluate


@torch.no_grad()
def evaluate_modality_combinations(
    model: torch.nn.Module, loader: Any, loss_fn: Any, device: torch.device
) -> dict[str, dict[str, float]]:
    """Evaluate every non-empty mask without mutating records or manifests."""
    results: dict[str, dict[str, float]] = {}
    for combination in all_non_empty_modality_combinations():
        mask = combination_to_mask(combination)
        masked_batches = []
        for original in loader:
            batch = dict(original)
            image = batch["image"].clone()
            available = mask.view(1, 4, *([1] * (image.ndim - 2))).to(image.device)
            batch["image"] = image * available.to(image.dtype)
            batch["modality_mask"] = mask.repeat(image.shape[0], 1)
            masked_batches.append(batch)
        metrics = evaluate(model, masked_batches, loss_fn, device)
        metrics["subjects"] = metrics.get("subjects", 0.0)
        metrics["available_modalities"] = float(mask.sum())
        metrics["counterfactual"] = 1.0
        results[modality_combination_key(combination)] = metrics
    return results


__all__ = ["evaluate", "evaluate_modality_combinations"]

__all__ = ["evaluate"]
