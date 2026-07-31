"""Single configurable MONAI U-Net baseline."""

from typing import Any

import torch


class MaskConcatUNet(torch.nn.Module):
    """U-Net baseline that spatially concatenates explicit modality presence masks."""

    supports_modality_mask = True
    uses_modality_specific_encoders = False
    aggregation_strategy = "fedavg"

    def __init__(self, network: torch.nn.Module, concat_mask: bool) -> None:
        super().__init__()
        self.network, self.concat_mask = network, concat_mask

    def forward(
        self, image: torch.Tensor, modality_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if image.shape[1] != 4:
            raise ValueError("BraTS baseline expects exactly four MRI image channels")
        if modality_mask is None:
            modality_mask = torch.ones((image.shape[0], 4), dtype=torch.bool, device=image.device)
        if modality_mask.shape != (image.shape[0], 4) or not modality_mask.any(dim=1).all():
            raise ValueError("each sample requires a non-empty [B, 4] modality mask")
        if self.concat_mask:
            shape = (image.shape[0], 4) + (1,) * (image.ndim - 2)
            presence = modality_mask.to(image.dtype).view(shape).expand_as(image)
            image = torch.cat((image, presence), dim=1)
        return self.network(image)


def create_unet(config: dict[str, Any]) -> MaskConcatUNet:
    from monai.networks.nets import UNet  # type: ignore[attr-defined]

    concat_mask = bool(config.get("concat_mask", False))
    network = UNet(
        spatial_dims=int(config.get("spatial_dims", 3)),
        in_channels=int(config.get("in_channels", 4)) + (4 if concat_mask else 0),
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
    return MaskConcatUNet(network, concat_mask)
