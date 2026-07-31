import pytest
import torch

from fl_brats.data.modalities import (
    ArbitraryMissingnessPolicy,
    CompleteMissingnessPolicy,
    all_non_empty_modality_combinations,
    combination_to_mask,
    combination_id_to_mask,
    mask_to_combination,
    mask_to_combination_id,
    modality_combination_counts,
    modality_counts,
)
from fl_brats.models.unet import MaskConcatUNet


def test_combinations_and_round_trip() -> None:
    assert len(all_non_empty_modality_combinations()) == 15
    assert mask_to_combination(combination_to_mask(("t1", "flair"))) == ("t1", "flair")
    mask = torch.tensor([[True, False, False, True], [False, True, True, False]])
    ids = mask_to_combination_id(mask)
    assert ids.tolist() == [9, 6]
    assert torch.equal(combination_id_to_mask(ids), mask)
    assert modality_counts(mask) == {"t1": 1, "t1ce": 1, "t2": 1, "flair": 1}
    assert modality_combination_counts(mask) == {6: 1, 9: 1}


def test_policies_are_stable_and_nonempty() -> None:
    policy = ArbitraryMissingnessPolicy(8, {"t1": 1, "t1ce": 1, "t2": 1, "flair": 1})
    first = policy.available_modalities(subject_id="s", client_id="c", split="train")
    second = policy.available_modalities(subject_id="s", client_id="c", split="train")
    assert first == second
    complete = CompleteMissingnessPolicy(2, {"client_0": ["t2", "flair"]})
    assert complete.available_modalities(subject_id="one", client_id="client_0", split="train") == (
        "t2",
        "flair",
    )


def test_masked_model_rejects_empty_mask() -> None:
    class Network(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value

    model = MaskConcatUNet(Network(), True)
    with pytest.raises(ValueError, match="non-empty"):
        model(torch.zeros(1, 4, 2, 2), torch.zeros(1, 4, dtype=torch.bool))
