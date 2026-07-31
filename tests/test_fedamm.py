import torch

from fl_brats.models.factory import create_model
from fl_brats.models.fedamm import FedAMMOutput, masked_mean, semantic_prototypes


def test_fedamm_forward_uses_typed_output_and_masks_missing_modalities() -> None:
    model = create_model({"name": "fedamm", "channels": [2, 4], "out_channels": 3})
    image = torch.randn(2, 4, 8, 8, 8)
    mask = torch.tensor([[True, False, False, False], [False, False, True, False]])
    output = model(image, mask)
    assert isinstance(output, FedAMMOutput)
    assert output.logits.shape == (2, 3, 8, 8, 8)
    assert output.combination_ids.tolist() == [1, 4]
    assert "t1ce" not in output.unimodal_features
    assert torch.count_nonzero(output.unimodal_features["t1"][1]) == 0


def test_masked_mean_handles_per_sample_combinations() -> None:
    mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
    values = {
        "t1": torch.tensor([[2.0], [2.0]]),
        "t1ce": torch.tensor([[10.0], [6.0]]),
    }
    assert torch.allclose(masked_mean(values, mask), torch.tensor([[2.0], [4.0]]))


def test_semantic_prototypes_pool_class_features() -> None:
    features = torch.ones(1, 2, 2, 2, 2)
    target = torch.zeros(1, 3, 2, 2, 2)
    target[:, 0] = 1
    prototypes = semantic_prototypes(features, target)
    assert prototypes.shape == (1, 3, 2)
    assert torch.allclose(prototypes[:, 0], torch.ones(1, 2))
