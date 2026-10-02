import torch

from fl_brats.models.factory import create_model
from fl_brats.models.fedamm import FedAMMOutput, masked_mean, semantic_prototypes
from fl_brats.training.engine import train_one_epoch


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


def test_fedamm_architecture_can_run_plain_missing_modality_fedavg_control() -> None:
    model = create_model(
        {
            "name": "fedamm",
            "channels": [2, 4],
            "out_channels": 3,
            "prototype_distillation_enabled": False,
            "global_prototype_alignment_enabled": False,
            "emit_prototypes": False,
            "modality_weighted_aggregation": False,
        }
    )
    loader = [
        {
            "image": torch.randn(2, 4, 8, 8, 8),
            "label": torch.zeros(2, 3, 8, 8, 8),
            "modality_mask": torch.tensor([[True, False, True, False], [False, True, False, True]]),
        }
    ]
    result = train_one_epoch(
        model,
        loader,  # type: ignore[arg-type]
        torch.nn.BCEWithLogitsLoss(),
        torch.optim.Adam(model.parameters(), lr=1e-4),
        torch.device("cpu"),
    )
    assert "fedamm_prototypes_json" not in result


def test_prototype_library_survives_checkpoint_round_trip() -> None:
    from fl_brats.federated.fedamm import GlobalPrototypeLibrary

    library = GlobalPrototypeLibrary()
    library.centres = {3: [[[0.5, 1.0]]], 15: [[[2.0, -1.0]]]}
    restored = GlobalPrototypeLibrary()
    restored.loads(library.dumps())
    assert restored.centres == library.centres
    assert restored.dumps() == library.dumps()


def test_loss_components_are_logged_without_changing_the_update() -> None:
    import json

    from fl_brats.models.fedamm import fedamm_auxiliary_loss

    torch.manual_seed(0)
    reference = create_model({"name": "fedamm", "channels": [2, 4], "out_channels": 3})
    logged = create_model({"name": "fedamm", "channels": [2, 4], "out_channels": 3})
    logged.load_state_dict(reference.state_dict())
    library = json.dumps({"5": [[[0.1] * 4] * 3], "10": [[[0.2] * 4] * 3]})
    reference.global_prototypes_json = library
    logged.global_prototypes_json = library
    label = (torch.rand(2, 3, 8, 8, 8) > 0.7).float()
    batch = {
        "image": torch.randn(2, 4, 8, 8, 8),
        "label": label,
        "modality_mask": torch.tensor([[True, False, True, False], [False, True, False, True]]),
    }
    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    output = reference(batch["image"], batch["modality_mask"])
    loss = loss_fn(output.logits, label) + fedamm_auxiliary_loss(output, label, 1.0)
    loss.backward()
    optimizer.step()

    result = train_one_epoch(
        logged,
        [batch],  # type: ignore[arg-type]
        loss_fn,
        torch.optim.SGD(logged.parameters(), lr=0.1),
        torch.device("cpu"),
    )
    for key, value in reference.state_dict().items():
        assert torch.equal(value, logged.state_dict()[key]), key
    assert result["train_loss"] == float(loss.detach())
    assert set(result) >= {"seg_loss", "distill_loss", "align_loss", "aux_loss_weighted", "aux_fraction"}
    assert {f"distill_loss_{m}" for m in ("t1", "t1ce", "t2", "flair")} <= set(result)
    terms = [result[f"distill_loss_{m}"] for m in ("t1", "t1ce", "t2", "flair")]
    assert abs(result["distill_loss"] - sum(terms) / 4) < 1e-6
    assert abs(result["aux_loss_weighted"] - (sum(terms) + result["align_loss"]) / 5) < 1e-6
    assert abs(result["seg_loss"] + result["aux_loss_weighted"] - result["train_loss"]) < 1e-6
    assert abs(result["aux_fraction"] - result["aux_loss_weighted"] / result["train_loss"]) < 1e-9


def test_plain_models_log_only_the_segmentation_loss() -> None:
    model = create_model(
        {
            "name": "fedamm",
            "channels": [2, 4],
            "out_channels": 3,
            "prototype_distillation_enabled": False,
            "global_prototype_alignment_enabled": False,
            "emit_prototypes": False,
        }
    )
    batch = {
        "image": torch.randn(2, 4, 8, 8, 8),
        "label": torch.zeros(2, 3, 8, 8, 8),
        "modality_mask": torch.ones(2, 4, dtype=torch.bool),
    }
    result = train_one_epoch(
        model,
        [batch],  # type: ignore[arg-type]
        torch.nn.BCEWithLogitsLoss(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.device("cpu"),
    )
    assert result["seg_loss"] == result["train_loss"]
    assert not {key for key in result if key.startswith(("distill", "align", "aux"))}
