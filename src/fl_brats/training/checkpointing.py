"""Safe CPU-portable checkpoint I/O."""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a dictionary")
    return payload


def validate_model_checkpoint(payload: dict[str, Any], model_config: dict[str, Any]) -> None:
    """Reject checkpoints whose explicit input strategy differs from the requested model."""
    saved = payload.get("model_config")
    mismatch = saved is not None and bool(saved.get("concat_mask", False)) != bool(
        model_config.get("concat_mask", False)
    )
    if mismatch:
        raise ValueError(
            "checkpoint concat_mask setting is incompatible with requested model configuration"
        )


def quantization_checkpoint_metadata(config: dict[str, Any], assignments: dict[str, dict[str, int]], residuals: dict[str, dict[str, torch.Tensor]] | None = None) -> dict[str, Any]:
    """Return explicit, CPU-portable state needed to resume compressed FedAvg."""
    return {
        "quantization": config,
        "quantization_assignments": assignments,
        "quantization_residuals": residuals or {},
        "quantization_schema_version": 1,
    }
