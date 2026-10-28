"""MONAI SegResNet backbone for the architecture comparison (RQ5).

Built inside the U-Net's wrapper so training, FedAvg aggregation, checkpoints
and evaluation treat it exactly as the U-Net; only the network changes. The
defaults follow Myronenko (2018), the BraTS 2018 winner: blocks_down
(1, 2, 2, 4), blocks_up (1, 1, 1) and group normalisation with 8 groups.
"""

from typing import Any

from .unet import MaskConcatUNet


def create_segresnet(config: dict[str, Any]) -> MaskConcatUNet:
    from monai.networks.nets import SegResNet  # type: ignore[attr-defined]

    concat_mask = bool(config.get("concat_mask", False))
    network = SegResNet(
        spatial_dims=int(config.get("spatial_dims", 3)),
        init_filters=int(config.get("init_filters", 16)),
        in_channels=int(config.get("in_channels", 4)) + (4 if concat_mask else 0),
        out_channels=int(config.get("out_channels", 3)),
        dropout_prob=config.get("dropout_prob"),
        blocks_down=tuple(config.get("blocks_down", [1, 2, 2, 4])),
        blocks_up=tuple(config.get("blocks_up", [1, 1, 1])),
        norm=tuple(config.get("norm", ["GROUP", {"num_groups": 8}])),
    )
    return MaskConcatUNet(network, concat_mask)
