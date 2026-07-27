"""Payload accounting for FP32 arrays and packed quantized update buffers."""

from typing import Any

import numpy as np

from .quantization import QuantizedClientUpdate, serialize_client_update


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


def quantized_payload_metrics(update: QuantizedClientUpdate) -> dict[str, Any]:
    """Measure actual application bytes from the exact serialized wire buffer."""
    serialized = serialize_client_update(update)
    scalar_count = sum(layer.num_parameters for layer in update.layers)
    logical_bits = sum(
        layer.num_parameters * layer.payload.params.bit_width for layer in update.layers
    )
    fp32 = scalar_count * 4
    return {
        "round": update.round_index,
        "client_id": update.client_id,
        "direction": "uplink",
        "uncompressed_fp32_payload_bytes": fp32,
        "logical_quantized_payload_bits": logical_bits,
        "actual_packed_payload_bytes": update.payload_bytes,
        "metadata_bytes": update.metadata_bytes,
        "total_serialized_bytes": len(serialized),
        "compression_ratio": fp32 / len(serialized) if serialized else float("inf"),
        "measurement_note": "application-level serialized bytes; transport headers are excluded",
    }
