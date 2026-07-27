"""Validation entry points."""

from typing import Any

from pydantic import ValidationError

from fl_brats.config.schema import ProjectConfig


def validate_config(values: dict[str, Any]) -> ProjectConfig:
    """Validate resolved Hydra values and return their typed representation."""
    return ProjectConfig.model_validate(values)


def format_validation_error(error: ValidationError) -> str:
    """Format Pydantic validation errors for command-line users."""
    return "\n".join(f"- {item['loc']}: {item['msg']}" for item in error.errors())
