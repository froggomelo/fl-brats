import torch

from fl_brats.federated.model_client import modality_count_metrics


def test_modality_count_metrics_stacks_tensor_masks() -> None:
    metrics = modality_count_metrics(
        [
            {"modality_mask": torch.tensor([True, False, True, False])},
            {"modality_mask": torch.tensor([True, True, False, False])},
        ]
    )

    assert metrics == {
        "fedamm_modality_count_t1": 2.0,
        "fedamm_modality_count_t1ce": 1.0,
        "fedamm_modality_count_t2": 1.0,
        "fedamm_modality_count_flair": 0.0,
    }
