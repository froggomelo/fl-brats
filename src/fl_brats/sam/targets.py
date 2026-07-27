"""BraTS overlapping binary region target helpers."""

import torch

REGIONS = ("WT", "TC", "ET")


def ensure_region_targets(target: torch.Tensor) -> torch.Tensor:
    if target.ndim < 4 or target.shape[1] != 3:
        raise ValueError("expected overlapping WT, TC, ET targets in channel order")
    return target.float()


def hierarchy_violations(logits_or_masks: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    masks = logits_or_masks > threshold if logits_or_masks.dtype.is_floating_point else logits_or_masks.bool()
    et_outside_tc = (masks[:, 2] & ~masks[:, 1]).sum().item()
    tc_outside_wt = (masks[:, 1] & ~masks[:, 0]).sum().item()
    total = max(int(masks[:, 0].numel()), 1)
    return {"et_outside_tc_rate": et_outside_tc / total, "tc_outside_wt_rate": tc_outside_wt / total}
