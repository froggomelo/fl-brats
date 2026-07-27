"""Multilabel Dice+BCE loss; never softmaxes overlapping BraTS regions."""

from typing import Any

from monai.losses import DiceCELoss  # type: ignore[attr-defined]


def create_loss(config: dict[str, Any]) -> DiceCELoss:
    return DiceCELoss(
        sigmoid=bool(config.get("sigmoid", True)),
        softmax=False,
        squared_pred=bool(config.get("squared_pred", False)),
        smooth_nr=float(config.get("smooth_nr", 1e-5)),
        smooth_dr=float(config.get("smooth_dr", 1e-5)),
        lambda_dice=float(config.get("dice_weight", 1.0)),
        lambda_ce=float(config.get("cross_entropy_weight", 1.0)),
        include_background=True,
        reduction=config.get("reduction", "mean"),
    )
