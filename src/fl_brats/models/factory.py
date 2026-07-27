"""Model construction and immutable architecture reporting."""

from typing import Any

import torch

from .unet import create_unet


def create_model(config: dict[str, Any]) -> torch.nn.Module:
    if config.get("name", "unet") != "unet":
        raise ValueError("only the baseline U-Net is supported")
    return create_unet(config)


def model_report(model: torch.nn.Module, config: dict[str, Any]) -> dict[str, Any]:
    state = model.state_dict()
    total = sum(p.numel() for p in model.parameters())
    return {
        "total_parameters": total,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_tensor_count": len(list(model.parameters())),
        "estimated_float32_model_size_bytes": total * 4,
        "state_dictionary_key_count": len(state),
        "model_config": config,
    }
