"""Authoritative BraTS segmentation metrics shared by every model family."""

from __future__ import annotations

import torch
from monai.metrics import compute_hausdorff_distance  # type: ignore[attr-defined]


REGIONS = ("wt", "tc", "et")


def _mask_edges(mask: torch.Tensor) -> torch.Tensor:
    """MONAI's edges of a [H, W, D] mask: the mask XOR its binary erosion.

    get_mask_edges erodes with scipy's default 6-connected cross and treats
    everything outside the (zero-padded) crop as background, so an edge voxel is
    a mask voxel with at least one of its six face neighbours outside the mask.
    Edges are translation-invariant, so the crop is not needed.
    """
    padded = torch.nn.functional.pad(mask[None, None].float(), (1, 1, 1, 1, 1, 1))[0, 0].bool()
    core = padded[1:-1, 1:-1, 1:-1]
    interior = (
        core
        & padded[:-2, 1:-1, 1:-1] & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1] & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2] & padded[1:-1, 1:-1, 2:]
    )
    return core & ~interior


def _nearest_edge_distances(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Euclidean distance from each ``source`` voxel to the nearest ``target`` voxel.

    Equals MONAI's exact EDT lookup. Squared distances come from a float32
    matrix product of integer coordinates: every product and sum is an integer
    below 2**24, so float32 is exact provided TF32 (which truncates the
    mantissa) is off, and it is forced off here. The square root is taken in
    float64 and stored as float32, as monai_distance_transform_edt does.
    """
    x = source.nonzero().float()
    y = target.nonzero().float()
    x2, y2 = (x * x).sum(1), (y * y).sum(1)
    best = torch.empty(len(x), dtype=torch.float32, device=source.device)
    chunk = max(1, (1 << 25) // max(len(y), 1))  # ~128 MB of float32 per block
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, len(x), chunk):
            block = x[start : start + chunk]
            squared = x2[start : start + chunk, None] + y2[None, :] - 2.0 * (block @ y.T)
            best[start : start + chunk] = squared.min(dim=1).values
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return best.double().sqrt().float()


def _hd95_gpu(predicted: torch.Tensor, truth: torch.Tensor) -> float:
    """HD95 of two non-empty [H, W, D] masks, identical to MONAI's on CPU."""
    edges_pred, edges_gt = _mask_edges(predicted), _mask_edges(truth)
    directed = (
        _nearest_edge_distances(edges_pred, edges_gt).cpu(),
        _nearest_edge_distances(edges_gt, edges_pred).cpu(),
    )
    return float(torch.max(torch.stack([torch.quantile(d, 0.95) for d in directed])))


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
            elif predicted.is_cuda:
                # MONAI's CPU distance transform takes seconds per region for
                # scattered predictions; the GPU edge search gives the same values.
                value = _hd95_gpu(predicted[0, 0], truth[0, 0])
            else:
                distance = compute_hausdorff_distance(
                    predicted.float(), truth.float(), include_background=True, percentile=95
                )
                value = float(distance.item())
            values[channel].append(value)
    return [sum(items) / len(items) if items else 0.0 for items in values]


def segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    metric_names: tuple[str, ...] = ("dice", "hd95", "sensitivity"),
) -> dict[str, float]:
    """Compute subject-macro Dice, sensitivity, and HD95 for WT, TC, and ET."""
    unsupported = set(metric_names) - {"dice", "hd95", "sensitivity"}
    if unsupported:
        raise ValueError(f"unsupported segmentation metrics: {sorted(unsupported)}")
    prediction = torch.sigmoid(logits) >= threshold
    truth = target.bool()
    result: dict[str, float] = {}
    hd95 = _hd95(prediction, truth) if "hd95" in metric_names else None
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
        if "dice" in metric_names:
            result[f"{region}_dice"] = float(dice.mean())
        if hd95 is not None:
            result[f"{region}_hd95"] = hd95[channel]
        if "sensitivity" in metric_names:
            sensitivity = torch.where(
                actual_count == 0,
                (predicted_count == 0).float(),
                intersection / actual_count,
            )
            result[f"{region}_sensitivity"] = float(sensitivity.mean())
    for metric in metric_names:
        result[f"mean_{metric}"] = sum(result[f"{region}_{metric}"] for region in REGIONS) / 3
    return result
