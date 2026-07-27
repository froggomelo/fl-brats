"""Validated ordered conversion between PyTorch state and Flower ndarrays."""

from collections import OrderedDict

import numpy as np
import torch


def state_keys(model: torch.nn.Module) -> list[str]:
    if hasattr(model, "federated_state_dict"):
        return list(model.federated_state_dict().keys())  # type: ignore[attr-defined,no-any-return]
    return list(model.state_dict().keys())


def state_to_arrays(model: torch.nn.Module) -> tuple[list[str], list[np.ndarray]]:
    state = model.federated_state_dict() if hasattr(model, "federated_state_dict") else model.state_dict()  # type: ignore[attr-defined]
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
    current = model.federated_state_dict() if hasattr(model, "federated_state_dict") else model.state_dict()  # type: ignore[attr-defined]
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, array in zip(keys, arrays, strict=True):
        if tuple(array.shape) != tuple(current[key].shape):
            raise ValueError(f"tensor shape mismatch for {key}")
        output[key] = torch.as_tensor(array, dtype=current[key].dtype)
    return output


def load_arrays(model: torch.nn.Module, keys: list[str], arrays: list[np.ndarray]) -> None:
    state = arrays_to_state(model, keys, arrays)
    if hasattr(model, "load_federated_state_dict"):
        model.load_federated_state_dict(state)  # type: ignore[attr-defined]
    else:
        model.load_state_dict(state, strict=True)
