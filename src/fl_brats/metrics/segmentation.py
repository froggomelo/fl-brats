"""Authoritative BraTS segmentation metrics shared by every model family."""

from __future__ import annotations

import torch
from monai.metrics import compute_hausdorff_distance  # type: ignore[attr-defined]


REGIONS = ("wt", "tc", "et")


def _hd95(prediction: torch.Tensor, target: torch.Tensor) -> list[float]:
    """Return finite per-region HD95 in voxel units with explicit empty-mask policy."""
    values: list[list[float]] = [[] for _ in REGIONS]
    spatial_diagonal = float(sum((size - 1) ** 2 for size in target.shape[2:]) ** 0.5)
    for subject in range(target.shape[0]):
        for channel in range(len(REGIONS)):
            predicted = prediction[subject : subject + 1, channel : channel + 1]
            truth = target[subject : subject + 1, channel : channel + 1]
            predicted_nonempty, truth_nonempty = bool(predicted.any()), bool(truth.any())
            if not predicted_nonempty and not truth_nonempty:
                value = 0.0
            elif not predicted_nonempty or not truth_nonempty:
                value = spatial_diagonal
            else:
                distance = compute_hausdorff_distance(
                    predicted.float(), truth.float(), include_background=True, percentile=95
                )
                value = float(distance.item())
            values[channel].append(value)
    return [sum(items) / len(items) if items else 0.0 for items in values]


def segmentation_metrics(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """Compute subject-macro Dice, sensitivity, and HD95 for WT, TC, and ET."""
    prediction = torch.sigmoid(logits) >= threshold
    truth = target.bool()
    result: dict[str, float] = {}
    hd95 = _hd95(prediction, truth)
    for channel, region in enumerate(REGIONS):
        predicted = prediction[:, channel].flatten(1)
        actual = truth[:, channel].flatten(1)
        intersection = (predicted & actual).sum(1).float()
        predicted_count, actual_count = predicted.sum(1).float(), actual.sum(1).float()
        dice_denominator = predicted_count + actual_count
        dice = torch.where(
            dice_denominator == 0,
            torch.ones_like(dice_denominator),
            2 * intersection / dice_denominator,
        )
        sensitivity = torch.where(
            actual_count == 0,
            (predicted_count == 0).float(),
            intersection / actual_count,
        )
        result[f"{region}_dice"] = float(dice.mean())
        result[f"{region}_hd95"] = hd95[channel]
        result[f"{region}_sensitivity"] = float(sensitivity.mean())
    for metric in ("dice", "hd95", "sensitivity"):
        result[f"mean_{metric}"] = sum(result[f"{region}_{metric}"] for region in REGIONS) / 3
    return result
