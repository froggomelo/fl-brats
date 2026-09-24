"""What a model checkpoint stores, and how it is loaded back.

Most models store their full state dictionary and load it strictly. A model
with a large frozen pretrained part (SAM) stores only its trainable weights
and rebuilds the frozen part from its verified pretrained checkpoint.
"""

from typing import Any

import torch


def checkpoint_state(model: torch.nn.Module) -> dict[str, Any]:
    if hasattr(model, "checkpoint_state_dict"):
        return dict(model.checkpoint_state_dict())
    return model.state_dict()


def load_checkpoint_state(model: torch.nn.Module, state: dict[str, Any]) -> None:
    if hasattr(model, "load_checkpoint_state_dict"):
        model.load_checkpoint_state_dict(state)
    else:
        model.load_state_dict(state, strict=True)
