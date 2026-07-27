"""Single configurable MONAI U-Net baseline."""

from typing import Any

from monai.networks.nets import UNet  # type: ignore[attr-defined]


def create_unet(config: dict[str, Any]) -> UNet:
    return UNet(
        spatial_dims=int(config.get("spatial_dims", 3)),
        in_channels=int(config.get("in_channels", 4)),
        out_channels=int(config.get("out_channels", 3)),
        channels=tuple(config.get("channels", [16, 32, 64, 128, 256])),
        strides=tuple(config.get("strides", [2, 2, 2, 2])),
        kernel_size=int(config.get("kernel_size", 3)),
        up_kernel_size=int(config.get("up_kernel_size", 3)),
        num_res_units=int(config.get("num_res_units", 2)),
        dropout=float(config.get("dropout", 0.0)),
        norm=config.get("normalisation", "INSTANCE"),
        act=config.get("activation", "PRELU"),
        bias=bool(config.get("bias", True)),
    )
