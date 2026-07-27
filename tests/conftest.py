from pathlib import Path

import pytest


@pytest.fixture
def roots(tmp_path: Path) -> dict[str, str]:
    return {
        name: str(tmp_path / name) for name in ("data", "outputs", "logs", "cache", "checkpoints")
    }
