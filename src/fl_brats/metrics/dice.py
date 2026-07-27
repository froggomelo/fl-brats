"""Per-subject multilabel Dice. Empty GT and empty prediction scores one."""

import torch


def dice_metrics(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    prediction = torch.sigmoid(logits) >= threshold
    target = target.bool()
    values: list[torch.Tensor] = []
    for channel in range(3):
        p, t = prediction[:, channel].flatten(1), target[:, channel].flatten(1)
        intersection, total = (p & t).sum(1).float(), p.sum(1).float() + t.sum(1).float()
        values.append(
            torch.where(total == 0, torch.ones_like(total), 2 * intersection / total).mean()
        )
    names = ("wt_dice", "tc_dice", "et_dice")
    result = {name: float(value) for name, value in zip(names, values, strict=True)}
    result["mean_dice"] = sum(result.values()) / 3
    return result
