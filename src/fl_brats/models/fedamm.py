"""FedAMM model: modality-specific 3D encoders with a shared decoder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fl_brats.data.modalities import MODALITIES, mask_to_combination_id, validate_modality_mask


@dataclass
class FedAMMOutput:
    logits: torch.Tensor
    fused_features: torch.Tensor
    fused_skip_features: list[torch.Tensor]
    unimodal_features: dict[str, torch.Tensor]
    unimodal_skip_features: dict[str, list[torch.Tensor]]
    modality_mask: torch.Tensor
    combination_ids: torch.Tensor


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=True),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.PReLU(),
        nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=True),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.PReLU(),
    )


class ModalityEncoder(nn.Module):
    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_block(1 if index == 0 else channels[index - 1], width) for index, width in enumerate(channels)]
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        value = image
        for index, block in enumerate(self.blocks):
            value = block(value)
            skips.append(value)
            if index != len(self.blocks) - 1:
                value = self.pool(value)
        return value, skips


class SharedDecoder(nn.Module):
    def __init__(self, channels: list[int], out_channels: int) -> None:
        super().__init__()
        pairs = list(zip(reversed(channels[1:]), reversed(channels[:-1]), strict=True))
        self.ups = nn.ModuleList([nn.ConvTranspose3d(src, dst, 2, stride=2) for src, dst in pairs])
        self.blocks = nn.ModuleList([_block(dst * 2, dst) for _, dst in pairs])
        self.head = nn.Conv3d(channels[0], out_channels, 1)

    def forward(self, bottleneck: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        value = bottleneck
        for level, (up, block) in enumerate(zip(self.ups, self.blocks, strict=True)):
            value = up(value)
            skip = skips[-(level + 2)]
            if value.shape[2:] != skip.shape[2:]:
                value = F.interpolate(value, size=skip.shape[2:], mode="trilinear", align_corners=False)
            value = block(torch.cat((value, skip), dim=1))
        return self.head(value)


class FedAMMUNet(nn.Module):
    """Practical FedAMM segmentation network for arbitrary per-sample missingness."""

    supports_modality_mask = True
    uses_modality_specific_encoders = True
    aggregation_strategy = "fedamm"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        channels = [int(x) for x in config.get("channels", [8, 16, 32, 64])]
        if len(channels) < 2:
            raise ValueError("FedAMM requires at least two channel levels")
        self.prototype_layer = int(config.get("prototype_layer", -1))
        self.prototype_temperature = float(config.get("prototype_temperature", 1.0))
        self.prototype_loss_weight = float(config.get("prototype_loss_weight", 1.0))
        self.prototype_distillation_enabled = bool(config.get("prototype_distillation_enabled", True))
        self.global_prototype_alignment_enabled = bool(config.get("global_prototype_alignment_enabled", True))
        self.emit_prototypes = bool(config.get("emit_prototypes", True))
        self.modality_weighted_aggregation = bool(config.get("modality_weighted_aggregation", True))
        self.modality_encoders = nn.ModuleDict({modality: ModalityEncoder(channels) for modality in MODALITIES})
        self.decoder = SharedDecoder(channels, int(config.get("out_channels", 3)))

    def forward(self, image: torch.Tensor, modality_mask: torch.Tensor | None = None) -> FedAMMOutput:
        if image.ndim != 5 or image.shape[1] != len(MODALITIES):
            raise ValueError("FedAMM expects image tensor [B, 4, H, W, D]")
        if modality_mask is None:
            modality_mask = torch.ones((image.shape[0], len(MODALITIES)), dtype=torch.bool, device=image.device)
        mask = validate_modality_mask(modality_mask.to(image.device))
        features: dict[str, torch.Tensor] = {}
        skips: dict[str, list[torch.Tensor]] = {}
        for index, modality in enumerate(MODALITIES):
            if not mask[:, index].any():
                continue
            feat, skip = self.modality_encoders[modality](image[:, index : index + 1])
            gate = mask[:, index].to(feat.dtype).view(-1, 1, 1, 1, 1)
            features[modality] = feat * gate
            skips[modality] = [item * gate for item in skip]
        fused = masked_mean(features, mask)
        fused_skips = [masked_mean({m: values[level] for m, values in skips.items()}, mask) for level in range(len(next(iter(skips.values()))))]
        output = FedAMMOutput(
            logits=self.decoder(fused, fused_skips),
            fused_features=fused,
            fused_skip_features=fused_skips,
            unimodal_features=features,
            unimodal_skip_features=skips,
            modality_mask=mask,
            combination_ids=mask_to_combination_id(mask),
        )
        setattr(output, "global_prototype_alignment_enabled", self.global_prototype_alignment_enabled)
        if hasattr(self, "global_prototypes_json"):
            setattr(output, "global_prototypes_json", getattr(self, "global_prototypes_json"))
        return output

    def federated_state_dict(self, scope: str = "trainable_only") -> dict[str, torch.Tensor]:
        return self.state_dict()

    def load_federated_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state, strict=True)

    def modality_parameter_prefixes(self) -> dict[str, str]:
        return {modality: f"modality_encoders.{modality}." for modality in MODALITIES}


def masked_mean(values: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    if not values:
        raise ValueError("cannot fuse an empty modality feature set")
    output = None
    denominator = None
    for index, modality in enumerate(MODALITIES):
        if modality not in values:
            continue
        gate = mask[:, index].to(values[modality].dtype).view(-1, *([1] * (values[modality].ndim - 1)))
        output = values[modality] * gate if output is None else output + values[modality] * gate
        denominator = gate if denominator is None else denominator + gate
    assert output is not None and denominator is not None
    return output / denominator.clamp_min(1.0)


def semantic_prototypes(features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pool class-specific feature prototypes from overlapping WT/TC/ET masks."""
    if features.shape[0] != target.shape[0]:
        raise ValueError("feature and target batch sizes differ")
    resized = F.interpolate(target.float(), size=features.shape[2:], mode="nearest")
    flat_features = features.flatten(2)
    flat_target = resized.flatten(2)
    numerator = torch.einsum("bfs,bcs->bcf", flat_features, flat_target)
    denominator = flat_target.sum(dim=2, keepdim=True).clamp_min(1.0)
    return numerator / denominator


def fedamm_auxiliary_loss(
    output: FedAMMOutput,
    target: torch.Tensor,
    prototype_weight: float = 1.0,
    *,
    distillation_enabled: bool = True,
) -> torch.Tensor:
    if prototype_weight <= 0:
        return output.logits.new_zeros(())
    losses = []
    if distillation_enabled:
        teacher = semantic_prototypes(output.fused_features, target).detach()
        for modality, features in output.unimodal_features.items():
            index = MODALITIES.index(modality)
            present = output.modality_mask[:, index]
            if present.any():
                student = semantic_prototypes(features[present], target[present])
                losses.append(F.mse_loss(student, teacher[present]))
    global_loss = global_prototype_alignment_loss(output, target)
    if global_loss is not None:
        losses.append(global_loss)
    if not losses:
        return output.logits.new_zeros(())
    return output.logits.new_tensor(float(prototype_weight)) * torch.stack(losses).mean()


def global_prototype_alignment_loss(output: FedAMMOutput, target: torch.Tensor) -> torch.Tensor | None:
    if not bool(getattr(output, "global_prototype_alignment_enabled", True)):
        return None
    payload = getattr(output, "global_prototypes_json", None)
    if not isinstance(payload, str) or not payload:
        return None
    library = json.loads(payload)
    local = semantic_prototypes(output.fused_features, target)
    losses = []
    for index, combo_id in enumerate(output.combination_ids.detach().cpu().tolist()):
        centres = library.get(str(int(combo_id)))
        if not centres:
            continue
        centre_tensor = torch.as_tensor(centres, dtype=local.dtype, device=local.device)
        distances = ((centre_tensor - local[index].unsqueeze(0)) ** 2).mean(dim=(1, 2))
        losses.append(distances.min())
    return torch.stack(losses).mean() if losses else None


def prototype_payload(output: FedAMMOutput, target: torch.Tensor) -> dict[str, Any]:
    proto = semantic_prototypes(output.fused_features.detach(), target.detach())
    return {
        "combination_ids": output.combination_ids.detach().cpu().tolist(),
        "prototypes": proto.detach().cpu().tolist(),
    }


def create_fedamm(config: dict[str, Any]) -> FedAMMUNet:
    return FedAMMUNet(config)
