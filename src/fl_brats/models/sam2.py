"""SAM 2 image-mode adapter (never enables video propagation by default)."""

from typing import Any

from .sam_adapter import SAMAdapter


class SAM2Adapter(SAMAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        if config.get("video_memory_propagation", False):
            raise ValueError("video-memory propagation is disabled for independent BraTS subjects")
        super().__init__(config, generation="sam2")
