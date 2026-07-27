"""SAM 1 adapter; checkpoint source/API are intentionally separate from SAM 2."""

from typing import Any

from .sam_adapter import SAMAdapter


class SAM1Adapter(SAMAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, generation="sam1")
