"""BraTS 2021 labels: WT={1,2,4}, TC={1,4}, ET={4}."""

import numpy as np
import torch


def brats_to_regions(label: np.ndarray | torch.Tensor, unexpected: str = "error") -> torch.Tensor:
    """Convert integer BraTS labels to overlapping WT, TC, ET float channels."""
    value = torch.as_tensor(label).long()
    observed = set(torch.unique(value).tolist()); unknown = observed - {0, 1, 2, 4}
    if unknown and unexpected == "error": raise ValueError(f"unexpected BraTS labels: {sorted(unknown)}")
    return torch.stack(((value > 0), ((value == 1) | (value == 4)), value == 4)).float()
