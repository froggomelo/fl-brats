import pytest
import torch

from fl_brats.data.modalities import (
    ArbitraryMissingnessPolicy,
    CompleteMissingnessPolicy,
    DirichletCombinationPolicy,
    dirichlet_combination_assignment,
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


def test_dirichlet_combination_gives_every_client_each_combination_once_at_least() -> None:
    clients = [[f"s{c}_{i:03d}" for i in range(n)] for c, n in enumerate((60, 30, 20, 16))]
    first, proportions = dirichlet_combination_assignment(clients, 0.1, 42)
    second, _ = dirichlet_combination_assignment(clients, 0.1, 42)
    assert first == second and len(proportions) == 4
    options = set(all_non_empty_modality_combinations())
    for subjects, assigned in zip(clients, first, strict=True):
        assert set(assigned) == set(subjects)
        assert set(assigned.values()) == options
    with pytest.raises(ValueError, match="at least 15"):
        dirichlet_combination_assignment([[f"s{i}" for i in range(14)]], 0.1, 42)


def test_dirichlet_policy_draws_held_out_masks_from_pooled_training_mix() -> None:
    policy = DirichletCombinationPolicy(42, 0.1, {"t1": 3, "t2+flair": 1})
    draws = [
        policy.available_modalities(subject_id=f"v{i}", client_id="global_evaluation", split="validation")
        for i in range(400)
    ]
    assert set(draws) == {("t1",), ("t2", "flair")}
    assert 0.65 < draws.count(("t1",)) / len(draws) < 0.85
    with pytest.raises(ValueError, match="partition manifests"):
        policy.available_modalities(subject_id="x", client_id="client_0", split="train")
