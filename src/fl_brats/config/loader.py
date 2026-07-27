"""Hydra configuration loading without changing the process working directory."""

from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def config_directory() -> Path:
    """Return the repository configuration directory."""
    return Path(__file__).resolve().parents[3] / "configs"


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose the root Hydra config, applying standard Hydra overrides."""
    with initialize_config_dir(version_base=None, config_dir=str(config_directory())):
        return compose(config_name="config", overrides=overrides or [])


def resolved_dict(config: DictConfig) -> dict[str, Any]:
    """Convert a composed configuration to fully resolved plain values."""
    value = OmegaConf.to_container(config, resolve=True)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)
