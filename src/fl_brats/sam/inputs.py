"""MRI slice intensity conversion for SAM-compatible image tensors."""

from typing import Any

import torch
import torch.nn.functional as F


def prepare_sam_image(image: torch.Tensor, config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Clip/scale per slice and resize without applying natural-image statistics."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("SAM image input must be [B, 3, H, W]")
    low, high = config.get("clip", [-5.0, 5.0])
    if high <= low:
        raise ValueError("clip maximum must exceed minimum")
    original = tuple(image.shape[-2:])
    image = image.clamp(float(low), float(high))
    image = (image - float(low)) / (float(high) - float(low))
    size = config.get("input_size", 1024)
    size = (size, size) if isinstance(size, int) else tuple(size)
    resized = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
    return resized, {"original_size": original, "resized_size": size, "clip": [low, high], "range": [0, 1]}
