"""SAM 1 and SAM 2 image-mode adapters for 2D BraTS slices.

Each adapter wraps an official pretrained model (``segment_anything`` ViT-B or
``sam2`` Hiera) whose image encoder stays frozen. Trainable, and federated:

* a 1x1 projection from the four MRI channels to the encoder's three,
  initialised to the fixed FLAIR/T1ce/T2 mapping so training starts from it;
* one learned sparse prompt token per region (WT, TC, ET), replacing clicks
  or boxes, so no label ever reaches the model at inference;
* the pretrained mask decoder, which turns each region token into a mask.

Slices are resized to ``input_size`` (512 by default; SAM's positional
embeddings are interpolated from 1024) with each model's own preprocessing,
and the low-resolution masks are mapped back to the slice's original size.
``backend: tiny`` builds a small random encoder for checkpoint-free tests.
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fl_brats.models.base import SegmentationModel
from fl_brats.sam.fine_tuning import configure_fine_tuning
from fl_brats.sam.modalities import CANONICAL_MODALITIES, LearnedFourToThree, fixed_rgb_mapping
from fl_brats.sam.prompts import LearnedRegionPrompts
from fl_brats.sam.validation import verify_pretrained

REGIONS = 3
SAM1_PIXEL_MEAN = (123.675, 116.28, 103.53)
SAM1_PIXEL_STD = (58.395, 57.12, 57.375)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SAM2_CONFIGS = {"sam2_hiera_tiny": "configs/sam2/sam2_hiera_t.yaml"}


def _autocast(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _interpolate_grid(values: torch.Tensor, size: int) -> torch.Tensor:
    """Resize a [1, H, W, C] positional grid to [1, size, size, C]."""
    resized = F.interpolate(
        values.permute(0, 3, 1, 2).float(), size=(size, size), mode="bicubic", align_corners=False
    )
    return resized.permute(0, 2, 3, 1).to(values.dtype)


class _Sam1Backend(nn.Module):
    """ViT image encoder, prompt encoder and mask decoder of segment_anything."""

    stride = 16

    def __init__(self, config: dict[str, Any], checkpoint: Path, input_size: int) -> None:
        super().__init__()
        from segment_anything import sam_model_registry

        variant = str(config.get("variant", "vit_b"))
        sam = sam_model_registry[variant](checkpoint=str(checkpoint))
        grid = input_size // self.stride
        encoder = sam.image_encoder
        if encoder.pos_embed is not None and encoder.pos_embed.shape[1] != grid:
            encoder.pos_embed = nn.Parameter(_interpolate_grid(encoder.pos_embed.data, grid))
        encoder.img_size = input_size
        # Global-attention blocks interpolate their relative positions to the
        # new token grid themselves (segment_anything get_rel_pos).
        sam.prompt_encoder.image_embedding_size = (grid, grid)
        sam.prompt_encoder.input_image_size = (input_size, input_size)
        self.image_encoder = encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        self.register_buffer("pixel_mean", torch.tensor(SAM1_PIXEL_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(SAM1_PIXEL_STD).view(1, 3, 1, 1), persistent=False)
        self.input_size = input_size

    def preprocess(self, image01: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Resize the longest side and pad bottom/right, as SAM's predictor does."""
        height, width = image01.shape[-2:]
        scale = self.input_size / max(height, width)
        resized_size = (int(height * scale + 0.5), int(width * scale + 0.5))
        image = F.interpolate(image01, size=resized_size, mode="bilinear", align_corners=False)
        image = (image * 255.0 - self.pixel_mean) / self.pixel_std
        pad = (0, self.input_size - resized_size[1], 0, self.input_size - resized_size[0])
        return F.pad(image, pad), resized_size

    def postprocess(self, masks: torch.Tensor, resized: tuple[int, int], original: tuple[int, int]) -> torch.Tensor:
        masks = F.interpolate(masks, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        masks = masks[..., : resized[0], : resized[1]]
        return F.interpolate(masks, size=original, mode="bilinear", align_corners=False)

    def encode(self, image: torch.Tensor) -> dict[str, Any]:
        return {"embedding": self.image_encoder(image)}

    def decode(self, features: dict[str, Any], tokens: torch.Tensor) -> torch.Tensor:
        """Batched SAM 1 mask prediction: one single-mask output per region token.

        Mirrors ``MaskDecoder.predict_masks`` with each image repeated once per
        region, since the official method assumes one image per call.
        """
        decoder = self.mask_decoder
        embedding = features["embedding"].repeat_interleave(REGIONS, dim=0)
        count = embedding.shape[0]
        output_tokens = torch.cat([decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0)
        tokens = torch.cat((output_tokens.unsqueeze(0).expand(count, -1, -1), tokens), dim=1)
        dense = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        src = embedding + dense
        pos_src = self.prompt_encoder.get_dense_pe().expand(count, -1, -1, -1)
        b, c, h, w = src.shape
        hs, src = decoder.transformer(src, pos_src, tokens)
        mask_token = hs[:, 1, :]
        upscaled = decoder.output_upscaling(src.transpose(1, 2).reshape(b, c, h, w))
        hyper = decoder.output_hypernetworks_mlps[0](mask_token)
        b, c, h, w = upscaled.shape
        return (hyper.unsqueeze(1) @ upscaled.view(b, c, h * w)).view(b, 1, h, w)


class _Sam2Backend(nn.Module):
    """Hiera image encoder, prompt encoder and mask decoder of sam2 (image mode)."""

    stride = 16

    def __init__(self, config: dict[str, Any], checkpoint: Path, input_size: int) -> None:
        super().__init__()
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from sam2.build_sam import _load_checkpoint

        variant = str(config.get("variant", "sam2_hiera_tiny"))
        if variant not in SAM2_CONFIGS:
            raise ValueError(f"unsupported SAM 2 variant {variant!r}; known: {sorted(SAM2_CONFIGS)}")
        # Instantiate from the packaged YAML directly: sam2's build_sam2 composes
        # through the global Hydra instance, which this project also uses.
        text = (resources.files("sam2") / SAM2_CONFIGS[variant]).read_text(encoding="utf-8")
        cfg = OmegaConf.create(text)
        cfg.model.image_size = input_size
        OmegaConf.resolve(cfg)
        sam2 = instantiate(cfg.model, _recursive_=True)
        _load_checkpoint(sam2, str(checkpoint))
        if not sam2.use_high_res_features_in_sam or not sam2.directly_add_no_mem_embed:
            raise ValueError("the SAM 2 image adapter expects high-resolution features and no-memory embedding")
        self.image_encoder = sam2.image_encoder
        self.prompt_encoder = sam2.sam_prompt_encoder
        self.mask_decoder = sam2.sam_mask_decoder
        self.no_mem_embed = sam2.no_mem_embed
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.input_size = input_size

    def preprocess(self, image01: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Square resize with ImageNet statistics, as SAM2Transforms does."""
        image = F.interpolate(image01, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return (image - self.pixel_mean) / self.pixel_std, (self.input_size, self.input_size)

    def postprocess(self, masks: torch.Tensor, resized: tuple[int, int], original: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(masks, size=original, mode="bilinear", align_corners=False)

    def encode(self, image: torch.Tensor) -> dict[str, Any]:
        fpn = list(self.image_encoder(image)["backbone_fpn"])
        # SAM2Base.forward_image and SAM2ImagePredictor.set_image, image mode.
        high_res = [self.mask_decoder.conv_s0(fpn[0]), self.mask_decoder.conv_s1(fpn[1])]
        embedding = fpn[-1] + self.no_mem_embed.view(1, -1, 1, 1)
        return {"embedding": embedding, "high_res": high_res}

    def decode(self, features: dict[str, Any], tokens: torch.Tensor) -> torch.Tensor:
        embedding = features["embedding"].repeat_interleave(REGIONS, dim=0)
        high_res = [feature.repeat_interleave(REGIONS, dim=0) for feature in features["high_res"]]
        dense = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand_as(embedding)
        masks, _, _, _ = self.mask_decoder(
            image_embeddings=embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=tokens,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res,
        )
        return masks


class _TinyBackend(nn.Module):
    """Random stand-in encoder/decoder so tests run without a checkpoint."""

    def __init__(self, config: dict[str, Any], input_size: int) -> None:
        super().__init__()
        width = int(config.get("test_encoder_width", 32))
        self.image_encoder = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.GELU())
        self.mask_decoder = nn.Conv2d(width, 1, kernel_size=1)
        self.token_projection = nn.Linear(256, width)
        self.input_size = input_size

    def preprocess(self, image01: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        size = (self.input_size, self.input_size)
        return F.interpolate(image01, size=size, mode="bilinear", align_corners=False), size

    def postprocess(self, masks: torch.Tensor, resized: tuple[int, int], original: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(masks, size=original, mode="bilinear", align_corners=False)

    def encode(self, image: torch.Tensor) -> dict[str, Any]:
        return {"embedding": self.image_encoder(image)}

    def decode(self, features: dict[str, Any], tokens: torch.Tensor) -> torch.Tensor:
        embedding = features["embedding"].repeat_interleave(REGIONS, dim=0)
        bias = self.token_projection(tokens.mean(dim=1))[:, :, None, None]
        return self.mask_decoder(embedding + bias)


class SAMAdapter(SegmentationModel):
    model_kind = "sam"
    supports_modality_mask = False

    def __init__(self, config: dict[str, Any], generation: str) -> None:
        super().__init__()
        self.config, self.generation = config, generation
        self.multimodal_adapter = str(config.get("multimodal_adapter", "learned_4_to_3"))
        self.federated_scope = str(config.get("federated_scope", "trainable_only"))
        self.input_size = int(config.get("input_size", 512))
        mapping = list(config.get("rgb_mapping", ["flair", "t1ce", "t2"]))
        if self.multimodal_adapter == "learned_4_to_3":
            self.projection: nn.Module = LearnedFourToThree()
            _initialise_as_mapping(self.projection.projection, mapping)
        elif self.multimodal_adapter == "fixed_rgb":
            self.projection = nn.Identity()
        else:
            raise ValueError("multimodal_adapter must be learned_4_to_3 or fixed_rgb")
        backend = str(config.get("backend", "pretrained"))
        if backend == "tiny":
            self.backend: nn.Module = _TinyBackend(config, self.input_size)
            self.pretrained: dict[str, Any] | None = None
        elif backend == "pretrained":
            checkpoint = config.get("checkpoint") or config.get("checkpoint_path")
            self.pretrained = verify_pretrained({**config, "checkpoint_path": checkpoint})
            builder = _Sam1Backend if generation == "sam1" else _Sam2Backend
            self.backend = builder(config, Path(self.pretrained["path"]), self.input_size)
        else:
            raise ValueError("backend must be pretrained or tiny")
        self.region_prompts = LearnedRegionPrompts(256, REGIONS)
        self.trainable = configure_fine_tuning(self, str(config.get("fine_tuning_mode", "adapter_decoder")))
        if any(name.startswith("backend.image_encoder.") for name in self.trainable):
            raise ValueError("the pretrained image encoder must stay frozen")
        self.slice_batch_size = int(config.get("slice_batch_size", 16))

    def train(self, mode: bool = True) -> SAMAdapter:
        super().train(mode)
        # The frozen encoder always runs in inference mode (no dropout/drop-path).
        self.backend.image_encoder.eval()
        return self

    def _rgb(self, slices: torch.Tensor) -> torch.Tensor:
        if self.multimodal_adapter == "fixed_rgb":
            return fixed_rgb_mapping(slices, self.config.get("rgb_mapping", ["flair", "t1ce", "t2"]))
        return self.projection(slices)

    def _to_unit_range(self, rgb: torch.Tensor) -> torch.Tensor:
        """Clip z-scored intensities and scale them to [0, 1]."""
        low, high = (float(value) for value in self.config.get("clip", [-5.0, 5.0]))
        if high <= low:
            raise ValueError("clip maximum must exceed minimum")
        return (rgb.clamp(low, high) - low) / (high - low)

    def forward(self, image: torch.Tensor, prompts: dict[str, Any] | None = None) -> torch.Tensor:
        """Map [N, 4, H, W] slices to [N, 3, H, W] region logits (WT, TC, ET)."""
        if image.ndim != 4 or image.shape[1] != 4:
            raise ValueError("SAM adapter expects 2D slices [N, 4, H, W]")
        if prompts:
            raise ValueError("SAM adapters use learned prompts only; external prompts are not accepted")
        return torch.cat([self._forward_chunk(chunk) for chunk in self._chunks(image)])

    def _chunks(self, image: torch.Tensor) -> Iterator[torch.Tensor]:
        size = self.slice_batch_size if not torch.is_grad_enabled() else len(image)
        for start in range(0, len(image), size):
            yield image[start : start + size]

    def _forward_chunk(self, slices: torch.Tensor) -> torch.Tensor:
        original = (int(slices.shape[-2]), int(slices.shape[-1]))
        image01 = self._to_unit_range(self._rgb(slices.float()))
        prepared, resized = self.backend.preprocess(image01)
        tokens = self.region_prompts(len(slices)).reshape(len(slices) * REGIONS, 1, -1)
        with _autocast(slices.device):
            features = self.backend.encode(prepared)
            masks = self.backend.decode(features, tokens.to(prepared.dtype))
        masks = masks.float().view(len(slices), REGIONS, *masks.shape[-2:])
        return self.backend.postprocess(masks, resized, original)

    # --------------------------------------------------------------- state
    def federated_state_dict(self, scope: str = "trainable_only") -> dict[str, torch.Tensor]:
        scope = self.federated_scope if scope == "trainable_only" else scope
        state = self.state_dict()
        allowed = set(state) if scope == "full_model" else set(self.trainable)
        return OrderedDict((name, value.detach().clone()) for name, value in state.items() if name in allowed)

    def load_federated_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = list(self.federated_state_dict())
        if list(state) != expected:
            raise ValueError("federated state names/order mismatch")
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"unexpected SAM state keys: {unexpected}")

    def checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        """What checkpoints store: the trainable weights only. The frozen
        pretrained part is rebuilt from its verified checkpoint file."""
        return self.federated_state_dict("trainable_only")

    def load_checkpoint_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = set(self.checkpoint_state_dict())
        if set(state) != expected:
            missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
            raise ValueError(f"SAM checkpoint state mismatch; missing={missing[:5]} extra={extra[:5]}")
        self.load_state_dict(state, strict=False)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super().checkpoint_metadata(),
            "generation": self.generation,
            "variant": self.config.get("variant"),
            "federated_scope": self.federated_scope,
            "pretrained": self.pretrained,
            "input_size": self.input_size,
        }


def _initialise_as_mapping(projection: nn.Conv2d, mapping: list[str]) -> None:
    """Start the learned projection as the fixed three-modality mapping."""
    with torch.no_grad():
        projection.weight.zero_()
        projection.bias.zero_()
        for output, name in enumerate(mapping):
            projection.weight[output, CANONICAL_MODALITIES.index(name), 0, 0] = 1.0
