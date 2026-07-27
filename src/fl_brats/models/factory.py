"""Model construction and immutable architecture reporting."""

from typing import Any

import torch

from .unet import create_unet
from .sam1 import SAM1Adapter
from .sam2 import SAM2Adapter
from .model_stats import model_statistics


def create_model(config: dict[str, Any]) -> torch.nn.Module:
    name = config.get("name", "unet")
    if name == "unet":
        return create_unet(config)
    if name == "sam1":
        return SAM1Adapter(config)
    if name == "sam2":
        return SAM2Adapter(config)
    raise ValueError("model name must be unet, sam1, or sam2")


def model_report(model: torch.nn.Module, config: dict[str, Any]) -> dict[str, Any]:
    state = model.state_dict()
    report = model_statistics(model)
    report.update({"parameter_tensor_count": len(list(model.parameters())), "estimated_float32_model_size_bytes": report["full_float32_bytes"], "state_dictionary_key_count": len(state), "model_config": config})
    return report
