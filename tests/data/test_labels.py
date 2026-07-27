import numpy as np
import pytest
import torch

from fl_brats.data.labels import brats_to_regions


def test_regions_are_binary_and_ordered() -> None:
    result = brats_to_regions(np.array([0, 1, 2, 4]))
    assert result.dtype == torch.uint8 and result.tolist() == [
        [0, 1, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
    ]


def test_unexpected_label_fails() -> None:
    with pytest.raises(ValueError):
        brats_to_regions(np.array([3]))
