"""Common, deliberately small interface for segmentation model adapters."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class SegmentationModel(torch.nn.Module, ABC):
    """Adapter contract; SAM inputs may carry prompts while U-Net accepts volumes."""

    model_kind: str = "segmentation"

    @abstractmethod
    def federated_state_dict(self, scope: str = "trainable_only") -> dict[str, torch.Tensor]:
        """Return the state deliberately transmitted by Flower."""

    @abstractmethod
    def load_federated_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load only the configured transmitted state."""

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {"model_kind": self.model_kind}
