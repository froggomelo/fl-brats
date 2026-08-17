import pytest
from pydantic import ValidationError

from fl_brats.config.loader import load_config, resolved_dict
from fl_brats.config.validation import validate_config


def test_config_loads() -> None:
    assert validate_config(resolved_dict(load_config())).precision == "float32"


def test_hydra_override() -> None:
    assert resolved_dict(load_config(["environment=debug"]))["environment"]["mode"] == "debug"


def test_initial_weights_override() -> None:
    path = "artifacts/initial_weights/fedamm_seed_42.pt"
    assert resolved_dict(load_config([f"initial_weights={path}"]))["initial_weights"] == path


def test_fedamm_control_overrides() -> None:
    config = resolved_dict(
        load_config(
            [
                "model=fedamm",
                "model.prototype_loss_weight=0.0",
                "model.prototype_distillation_enabled=false",
                "model.global_prototype_alignment_enabled=false",
                "model.emit_prototypes=false",
                "model.modality_weighted_aggregation=false",
            ]
        )
    )
    assert config["model"]["prototype_loss_weight"] == 0.0
    assert config["model"]["prototype_distillation_enabled"] is False
    assert config["model"]["global_prototype_alignment_enabled"] is False
    assert config["model"]["emit_prototypes"] is False
    assert config["model"]["modality_weighted_aggregation"] is False


@pytest.mark.parametrize(
    "override", ["precision=wrong", "num_workers=-1", "environment.server_port=80"]
)
def test_invalid_values(override: str) -> None:
    with pytest.raises(ValidationError):
        validate_config(resolved_dict(load_config([override])))


def test_missing_apptainer_image_is_schema_valid() -> None:
    assert validate_config(
        resolved_dict(load_config(["environment=apptainer"]))
    ).environment.image_path


def test_invalid_cluster_scheduler() -> None:
    with pytest.raises(ValidationError):
        validate_config(
            resolved_dict(load_config(["environment=cluster", "environment.scheduler=none"]))
        )
