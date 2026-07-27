"""Validated ordered conversion between PyTorch state and Flower ndarrays."""

from collections import OrderedDict

import numpy as np
import torch


def state_keys(model: torch.nn.Module) -> list[str]:
    return list(model.state_dict().keys())


def state_to_arrays(model: torch.nn.Module) -> tuple[list[str], list[np.ndarray]]:
    state = model.state_dict()
    keys = list(state)
    return keys, [state[key].detach().cpu().numpy().copy() for key in keys]


def arrays_to_state(
    model: torch.nn.Module, keys: list[str], arrays: list[np.ndarray]
) -> OrderedDict[str, torch.Tensor]:
    expected = state_keys(model)
    if keys != expected:
        raise ValueError("state-dictionary key ordering mismatch")
    if len(arrays) != len(expected):
        raise ValueError("tensor count mismatch")
    current = model.state_dict()
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, array in zip(keys, arrays, strict=True):
        if tuple(array.shape) != tuple(current[key].shape):
            raise ValueError(f"tensor shape mismatch for {key}")
        output[key] = torch.as_tensor(array, dtype=current[key].dtype)
    return output


def load_arrays(model: torch.nn.Module, keys: list[str], arrays: list[np.ndarray]) -> None:
    model.load_state_dict(arrays_to_state(model, keys, arrays), strict=True)
