"""Stable SHA-256 helpers used by reproducible data artefacts."""

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size_mb: int = 8) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size_mb * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
