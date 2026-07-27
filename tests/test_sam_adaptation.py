import pytest
import torch

from fl_brats.federated.parameters import load_arrays, state_to_arrays
from fl_brats.models.factory import create_model
from fl_brats.sam.modalities import fixed_rgb_mapping
from fl_brats.sam.prompts import generate_prompts
from fl_brats.sam.reconstruction import reconstruct_volume
from fl_brats.sam.slicing import extract_slices


def test_fixed_mapping_and_projection_preserve_expected_channels() -> None:
    image = torch.arange(4.0).view(1, 4, 1, 1).expand(1, 4, 2, 2)
    assert fixed_rgb_mapping(image)[0, :, 0, 0].tolist() == [3.0, 1.0, 2.0]
    model = create_model({"name": "sam1", "input_size": 8})
    assert model(torch.randn(2, 4, 5, 7)).shape == (2, 3, 5, 7)


def test_slice_metadata_and_reconstruction() -> None:
    image, target = torch.zeros(4, 2, 3, 4), torch.zeros(3, 2, 3, 4)
    target[:, :, :, 1] = 1
    slices = list(extract_slices(image, target, "subject", "axial"))
    assert [record.slice_index for _, _, record in slices] == list(range(4))
    output = reconstruct_volume([(index, torch.full((3, 2, 3), index)) for index in range(4)], (2, 3, 4))
    assert output.shape == (3, 2, 3, 4)
    assert output[:, :, :, 3].eq(3).all()


def test_primary_prompt_guard_and_trainable_state_round_trip() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        generate_prompts(torch.zeros(1, 3, 2, 2), {}, {}, torch.zeros(1, 3, 2, 2))
    model = create_model({"name": "sam2", "input_size": 8})
    keys, arrays = state_to_arrays(model)
    assert all("image_encoder" not in key for key in keys)
    load_arrays(model, keys, arrays)
