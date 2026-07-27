"""Named federated parameter helpers and base-checkpoint compatibility checks."""

from typing import Any

import torch


def federated_parameter_names(model: torch.nn.Module) -> list[str]:
    return list(model.federated_state_dict()) if hasattr(model, "federated_state_dict") else list(model.state_dict())


def verify_client_base_hash(server_hash: str | None, client_metadata: dict[str, Any]) -> None:
    client_hash = client_metadata.get("base_checkpoint_sha256")
    if server_hash and client_hash != server_hash:
        raise ValueError("client base checkpoint hash does not match server checkpoint")
