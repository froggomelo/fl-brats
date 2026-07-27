"""Shared adapter plumbing for SAM 1 and SAM 2 image-mode segmentation.

The encoder wrapper is intentionally API-neutral: production deployments inject an
official SAM encoder, while the compact encoder keeps tests checkpoint-free.
"""

from collections import OrderedDict
from typing import Any

import torch
from torch import nn

from fl_brats.models.base import SegmentationModel
from fl_brats.sam.fine_tuning import configure_fine_tuning
from fl_brats.sam.inputs import prepare_sam_image
from fl_brats.sam.modalities import LearnedFourToThree, fixed_rgb_mapping
from fl_brats.sam.prompts import LearnedRegionPrompts


class SAMAdapter(SegmentationModel):
    model_kind = "sam"
    supports_modality_mask = False

    def __init__(self, config: dict[str, Any], generation: str) -> None:
        super().__init__()
        self.config, self.generation = config, generation
        self.multimodal_adapter = str(config.get("multimodal_adapter", "learned_4_to_3"))
        self.federated_scope = str(config.get("federated_scope", "trainable_only"))
        if self.multimodal_adapter == "learned_4_to_3":
            self.projection: nn.Module = LearnedFourToThree()
        elif self.multimodal_adapter == "fixed_rgb":
            self.projection = nn.Identity()
        else:
            raise ValueError("multimodal_adapter must be learned_4_to_3 or fixed_rgb")
        width = int(config.get("test_encoder_width", 32))
        # Production adapter points may replace this module with official encoders.
        self.image_encoder = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.GELU())
        self.prompt_encoder = LearnedRegionPrompts(width)
        self.mask_decoder = nn.Conv2d(width, 3, kernel_size=1)
        configure_fine_tuning(self, str(config.get("fine_tuning_mode", "adapter_decoder")))

    def _rgb(self, slices: torch.Tensor) -> torch.Tensor:
        if self.multimodal_adapter == "fixed_rgb":
            return fixed_rgb_mapping(slices, self.config.get("rgb_mapping", ["flair", "t1ce", "t2"]))
        return self.projection(slices)

    def forward(self, image: torch.Tensor, prompts: dict[str, Any] | None = None) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("SAM adapter expects 2D slices [B,4,H,W]")
        rgb = self._rgb(image)
        prepared, metadata = prepare_sam_image(rgb, self.config)
        features = self.image_encoder(prepared)
        embedding = self.prompt_encoder(features.shape[0]).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
        logits = self.mask_decoder(features) + embedding
        return torch.nn.functional.interpolate(logits, size=metadata["original_size"], mode="bilinear", align_corners=False)

    def federated_state_dict(self, scope: str = "trainable_only") -> dict[str, torch.Tensor]:
        scope = self.federated_scope if scope == "trainable_only" else scope
        state = self.state_dict()
        allowed = set(state) if scope == "full_model" else {n for n, p in self.named_parameters() if p.requires_grad}
        return OrderedDict((name, value.detach().clone()) for name, value in state.items() if name in allowed)

    def load_federated_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = list(self.federated_state_dict())
        if list(state) != expected:
            raise ValueError("federated state names/order mismatch")
        self.load_state_dict(state, strict=False)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {**super().checkpoint_metadata(), "generation": self.generation, "variant": self.config.get("variant"), "federated_scope": self.federated_scope, "checkpoint_sha256": self.config.get("checkpoint_sha256")}
