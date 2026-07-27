"""Comparable full and transmitted parameter accounting."""

from typing import Any

import torch


def model_statistics(model: torch.nn.Module) -> dict[str, Any]:
    params = list(model.named_parameters())
    total = sum(p.numel() for _, p in params)
    trainable = sum(p.numel() for _, p in params if p.requires_grad)
    federated = model.federated_state_dict() if hasattr(model, "federated_state_dict") else model.state_dict()
    scalar_count = sum(value.numel() for value in federated.values())
    def group(token: str) -> int:
        return sum(p.numel() for name, p in params if token in name)
    return {"total_parameters": total, "trainable_parameters": trainable, "frozen_parameters": total - trainable, "percent_trainable": 100 * trainable / total if total else 0.0, "full_float32_bytes": total * 4, "trainable_parameter_bytes": trainable * 4, "federated_tensor_count": len(federated), "federated_scalar_count": scalar_count, "federated_payload_bytes": sum(v.detach().cpu().numpy().nbytes for v in federated.values()), "image_encoder_parameters": group("image_encoder"), "prompt_encoder_parameters": group("prompt_encoder"), "mask_decoder_parameters": group("mask_decoder"), "adapter_parameters": group("adapter"), "multimodal_projection_parameters": group("projection")}
