"""Payload accounting: byte counts are model payloads, not network traffic."""

from typing import Any

import numpy as np


def payload_metrics(
    arrays: list[np.ndarray], round_number: int, client_id: int, direction: str
) -> dict[str, Any]:
    scalars = sum(a.size for a in arrays)
    bytes_ = sum(a.nbytes for a in arrays)
    return {
        "round": round_number,
        "client_id": client_id,
        "direction": direction,
        "tensor_count": len(arrays),
        "scalar_parameter_count": scalars,
        "theoretical_float32_bytes": scalars * 4,
        "numpy_array_bytes": bytes_,
        "serialized_payload_bytes": None,
        "measurement_note": "payload bytes; exact network-interface traffic is not measured",
    }
