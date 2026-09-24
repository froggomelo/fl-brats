import os
from pathlib import Path

import pytest
import torch

from fl_brats.federated.parameters import load_arrays, state_to_arrays
from fl_brats.models.factory import create_model
from fl_brats.models.state import checkpoint_state, load_checkpoint_state
from fl_brats.sam.modalities import fixed_rgb_mapping
from fl_brats.sam.prompts import generate_prompts
from fl_brats.sam.reconstruction import reconstruct_volume
from fl_brats.sam.slicing import extract_slices
from fl_brats.training.engine import forward_batch, sample_training_slices, train_one_epoch

TINY = {"backend": "tiny", "input_size": 8}


def test_fixed_mapping_and_projection_preserve_expected_channels() -> None:
    image = torch.arange(4.0).view(1, 4, 1, 1).expand(1, 4, 2, 2)
    assert fixed_rgb_mapping(image)[0, :, 0, 0].tolist() == [3.0, 1.0, 2.0]
    model = create_model({"name": "sam1", **TINY})
    assert model(torch.randn(2, 4, 5, 7)).shape == (2, 3, 5, 7)


def test_learned_projection_starts_as_the_fixed_mapping() -> None:
    model = create_model({"name": "sam1", **TINY})
    image = torch.randn(2, 4, 3, 3)
    assert torch.equal(model._rgb(image), fixed_rgb_mapping(image))


def test_pretrained_backend_requires_a_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        create_model({"name": "sam1", "input_size": 8})


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
    model = create_model({"name": "sam2", **TINY})
    keys, arrays = state_to_arrays(model)
    assert all("image_encoder" not in key for key in keys)
    load_arrays(model, keys, arrays)


def test_checkpoint_state_holds_only_trainable_weights_and_round_trips() -> None:
    torch.manual_seed(0)
    model = create_model({"name": "sam1", **TINY})
    state = checkpoint_state(model)
    assert state and all("image_encoder" not in key for key in state)
    assert set(state) == {name for name, p in model.named_parameters() if p.requires_grad}
    torch.manual_seed(1)
    other = create_model({"name": "sam1", **TINY})
    load_checkpoint_state(other, state)
    for key, value in state.items():
        assert torch.equal(other.state_dict()[key], value)
    with pytest.raises(ValueError, match="mismatch"):
        load_checkpoint_state(other, {**state, "backend.image_encoder.0.weight": torch.zeros(1)})


def test_training_slices_favour_tumour_and_keep_image_label_pairs() -> None:
    torch.manual_seed(0)
    image = torch.zeros(1, 4, 6, 6, 10)
    image[..., 2:9] = 1.0  # brain in slices 2-8
    image[..., 5] = 5.0  # marks the only tumour slice
    label = torch.zeros(1, 3, 6, 6, 10)
    label[:, :, 1:3, 1:3, 5] = 1
    batch = sample_training_slices({"image": image, "label": label}, count=8, tumour_fraction=0.5)
    assert batch["image"].shape == (8, 4, 6, 6)
    assert batch["label"].shape == (8, 3, 6, 6)
    tumour = batch["label"].flatten(1).amax(dim=1) > 0
    assert int(tumour.sum()) >= 4
    # Every sampled slice has brain, and labels travel with their own image slice.
    assert (batch["image"].flatten(1).amax(dim=1) > 0).all()
    assert torch.equal(tumour, batch["image"][:, 0, 0, 0] == 5.0)


def test_slice_model_trains_on_slices_and_predicts_whole_volumes() -> None:
    torch.manual_seed(0)
    model = create_model({"name": "sam1", **TINY, "train_slices_per_subject": 4, "slice_batch_size": 3})
    image = torch.randn(1, 4, 6, 7, 5)
    label = (torch.rand(1, 3, 6, 7, 5) > 0.7).float()
    before = model.projection.projection.weight.detach().clone()
    loader = [{"image": image, "label": label}]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    result = train_one_epoch(model, loader, torch.nn.BCEWithLogitsLoss(), optimizer, torch.device("cpu"))
    assert result["samples"] == 4.0
    # Gradients reach the projection through the frozen encoder, which is unchanged.
    assert not torch.equal(before, model.projection.projection.weight)
    assert all(not p.requires_grad for p in model.backend.image_encoder.parameters())
    with torch.no_grad():
        model.eval()
        logits = forward_batch(model, {"image": image})
    assert logits.shape == (1, 3, 6, 7, 5)


PRETRAINED = {
    "sam1": ("vit_b", "artifacts/pretrained/sam_vit_b_01ec64.pth"),
    "sam2": ("sam2_hiera_tiny", "artifacts/pretrained/sam2_hiera_tiny.pt"),
}


@pytest.mark.integration
@pytest.mark.parametrize("generation", sorted(PRETRAINED))
def test_pretrained_adapter_at_512(generation: str) -> None:
    variant, checkpoint = PRETRAINED[generation]
    if not Path(checkpoint).is_file() or os.environ.get("FL_BRATS_SKIP_PRETRAINED"):
        pytest.skip(f"{checkpoint} is not available")
    pytest.importorskip("segment_anything" if generation == "sam1" else "sam2")
    torch.manual_seed(0)
    model = create_model({"name": generation, "variant": variant, "checkpoint": checkpoint, "input_size": 512})
    trainable = checkpoint_state(model)
    assert trainable and all("image_encoder" not in key for key in trainable)
    image = torch.randn(2, 4, 150, 170)
    logits = model(image)
    assert logits.shape == (2, 3, 150, 170)
    assert torch.isfinite(logits).all()
    logits.mean().backward()
    assert model.projection.projection.weight.grad is not None
    assert all(p.grad is None for p in model.backend.image_encoder.parameters())
