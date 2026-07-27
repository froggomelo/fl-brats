"""Non-leaking prompt construction for automated SAM experiments."""

from typing import Any

import torch
from torch import nn


class LearnedRegionPrompts(nn.Module):
    def __init__(self, embedding_dim: int = 256, regions: int = 3) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(torch.zeros(regions, embedding_dim))
        nn.init.normal_(self.embeddings, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.embeddings.unsqueeze(0).expand(batch_size, -1, -1)


def generate_prompts(images: torch.Tensor, metadata: dict[str, Any] | None, config: dict[str, Any], labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Only oracle diagnostic mode may use labels; primary modes reject them."""
    oracle = bool(config.get("oracle_prompt", False))
    if labels is not None and not oracle:
        raise ValueError("label-derived prompts are forbidden unless oracle_prompt=true")
    if oracle:
        if labels is None:
            raise ValueError("oracle prompts require labels")
        # Boxes are intentionally only a diagnostic facility.
        return {"oracle_masks": labels}
    return {"images": images, "metadata": metadata or {}}  # permitted inputs only
