"""Validation guards for checkpoint and prompt safety."""

from pathlib import Path
from typing import Any

from fl_brats.data.hashing import sha256_file


def verify_pretrained(config: dict[str, Any]) -> dict[str, Any]:
    path_value = config.get("checkpoint_path")
    if not path_value:
        raise ValueError("checkpoint_path is required")
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise ValueError(f"pretrained checkpoint is missing: {path}")
    actual = sha256_file(path)
    expected = config.get("checkpoint_sha256")
    if expected and actual.lower() != str(expected).lower():
        raise ValueError("pretrained checkpoint SHA-256 mismatch")
    if not config.get("variant"):
        raise ValueError("SAM model variant is required")
    return {"path": str(path), "sha256": actual, "variant": config["variant"], "source_url": config.get("source_url")}


def validate_primary_prompt(config: dict[str, Any]) -> None:
    if config.get("oracle_prompt", False):
        raise ValueError("oracle prompts are excluded from primary architecture comparisons")
