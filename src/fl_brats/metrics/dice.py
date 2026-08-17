"""Per-subject multilabel Dice. Empty GT and empty prediction scores one."""

import torch

from .segmentation import segmentation_metrics


def dice_metrics(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    metrics = segmentation_metrics(logits, target, threshold)
    return {key: value for key, value in metrics.items() if key.endswith("dice")}
