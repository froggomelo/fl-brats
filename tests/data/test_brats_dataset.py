import torch

from fl_brats.data.brats import BraTSDataset
from fl_brats.data.modalities import MODALITIES, combination_to_mask


def _record() -> dict[str, object]:
    return {
        "image": "unused",
        "label": "unused",
        "subject_id": "BraTS2021_00000",
        "client_id": "centralized",
        "available_modalities": MODALITIES,
        "modality_mask": combination_to_mask(MODALITIES),
    }


def test_dataset_attaches_metadata_to_transform_sample_lists() -> None:
    def transform(_: dict[str, object]) -> list[dict[str, torch.Tensor]]:
        return [
            {"image": torch.zeros(4, 3, 3, 3), "label": torch.zeros(3, 3, 3)},
            {"image": torch.ones(4, 3, 3, 3), "label": torch.zeros(3, 3, 3)},
        ]

    samples = BraTSDataset([_record()], transform=transform)[0]

    assert isinstance(samples, list)
    assert len(samples) == 2
    assert all(sample["subject_id"] == "BraTS2021_00000" for sample in samples)
    assert all(sample["client_id"] == "centralized" for sample in samples)
    assert all(torch.equal(sample["modality_mask"], torch.ones(4, dtype=torch.bool)) for sample in samples)


def test_dataset_restores_missing_modality_channels_for_sample_lists() -> None:
    mask = combination_to_mask(("t1", "flair"))
    record = _record() | {"available_modalities": ("t1", "flair"), "modality_mask": mask}

    def transform(_: dict[str, object]) -> list[dict[str, torch.Tensor]]:
        return [{"image": torch.ones(2, 2, 2, 2), "label": torch.zeros(2, 2, 2)}]

    sample = BraTSDataset([record], transform=transform)[0][0]

    assert sample["image"].shape == (4, 2, 2, 2)
    assert torch.equal(sample["image"][0], torch.ones(2, 2, 2))
    assert torch.equal(sample["image"][1], torch.zeros(2, 2, 2))
    assert torch.equal(sample["image"][2], torch.zeros(2, 2, 2))
    assert torch.equal(sample["image"][3], torch.ones(2, 2, 2))
