"""Quantized server-to-client broadcast (downlink).

The server and every client hold the same reconstructed model, the view. The
first broadcast of a process (round 1, or the first round after a resume) is the
full FP32 model and both sides set view = model. Later broadcasts carry
Q(global - view): each tensor of the difference is quantized symmetrically per
tensor at ``bit_width`` bits, and both sides add the dequantized difference to
their view. What the quantizer drops stays in global - view and is sent in later
rounds, so the error does not accumulate (error feedback without client state).
Clients train from the view; the server keeps the FP32 global model, which is
what is validated and checkpointed. Client deltas are taken against the view and
applied to the FP32 global model, as for the uplink.

Wire format (one uint8 array): ``FLDL`` + uint32 header length + JSON header
(round, bit width, per-tensor name, shape and scale) + packed codes + SHA-256.
Codes are packed least-significant-code first, as in the uplink serializer.
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Sequence

import numpy as np
import torch

from .quantization import integer_range, quantize_tensor

MAGIC = b"FLDL"
SUPPORTED_BIT_WIDTHS = (2, 4, 8)


def pack_codes(unsigned: np.ndarray, bit_width: int) -> bytes:
    """Pack unsigned codes (< 2**bit_width), least-significant code first."""
    per_byte = 8 // bit_width
    padded = np.zeros(-(-unsigned.size // per_byte) * per_byte, dtype=np.uint8)
    padded[: unsigned.size] = unsigned
    grouped = padded.reshape(-1, per_byte).astype(np.uint16)
    shifts = (np.arange(per_byte, dtype=np.uint16) * bit_width)[None, :]
    return (grouped << shifts).sum(axis=1).astype(np.uint8).tobytes()


def unpack_codes(payload: bytes, bit_width: int, count: int) -> np.ndarray:
    """Invert :func:`pack_codes`."""
    per_byte = 8 // bit_width
    raw = np.frombuffer(payload, dtype=np.uint8)
    if raw.size != -(-count // per_byte):
        raise ValueError("downlink payload length does not match its value count")
    shifts = (np.arange(per_byte, dtype=np.uint8) * bit_width)[None, :]
    mask = np.uint8((1 << bit_width) - 1)
    return ((raw[:, None] >> shifts) & mask).reshape(-1)[:count]


def is_downlink_payload(parameters: Sequence[np.ndarray]) -> bool:
    return (
        len(parameters) == 1
        and parameters[0].dtype == np.uint8
        and parameters[0].ndim == 1
        and parameters[0][:4].tobytes() == MAGIC
    )


def _decode(blob: bytes) -> tuple[int, list[tuple[str, np.ndarray]]]:
    """Return the round and the dequantized difference of each tensor."""
    if blob[:4] != MAGIC or len(blob) < 8 + 64:
        raise ValueError("not a downlink payload")
    header_len = struct.unpack("<I", blob[4:8])[0]
    header, body, digest = blob[8 : 8 + header_len], blob[8 + header_len : -64], blob[-64:]
    if hashlib.sha256(header + body).hexdigest().encode() != digest:
        raise ValueError("downlink payload checksum mismatch")
    meta = json.loads(header)
    bit_width = int(meta["bit_width"])
    qmin, _ = integer_range(bit_width, "symmetric")
    offset, layers = 0, []
    for item in meta["layers"]:
        count = int(np.prod(item["shape"], dtype=np.int64))
        length = -(-count // (8 // bit_width))
        codes = unpack_codes(body[offset : offset + length], bit_width, count).astype(np.int32) + qmin
        offset += length
        # The reconstruction rule of quantization.dequantize_tensor: code * scale.
        delta = codes.astype(np.float32) * np.float32(item["scale"])
        layers.append((item["name"], delta.reshape(item["shape"])))
    if offset != len(body):
        raise ValueError("downlink payload contains trailing bytes")
    return int(meta["round"]), layers


def _apply(view: list[np.ndarray], keys: Sequence[str], layers: list[tuple[str, np.ndarray]]) -> list[np.ndarray]:
    if [name for name, _ in layers] != list(keys):
        raise ValueError("downlink payload tensors do not match the model")
    return [(base + delta.astype(base.dtype)).astype(base.dtype) for base, (_, delta) in zip(view, layers)]


class DownlinkEncoder:
    """Server side: turn each round's global model into the broadcast payload."""

    def __init__(self, keys: Sequence[str], bit_width: int) -> None:
        if bit_width not in SUPPORTED_BIT_WIDTHS:
            raise ValueError(f"downlink bit width must be one of {SUPPORTED_BIT_WIDTHS}")
        self.keys, self.bit_width = list(keys), int(bit_width)
        self.view: list[np.ndarray] | None = None

    def encode(self, arrays: Sequence[np.ndarray], round_index: int) -> list[np.ndarray]:
        if any(not np.issubdtype(item.dtype, np.floating) for item in arrays):
            raise ValueError("downlink quantization requires a floating-point model state")
        if self.view is None:
            self.view = [np.array(item, copy=True) for item in arrays]
            return [np.array(item, copy=True) for item in arrays]
        qmin, _ = integer_range(self.bit_width, "symmetric")
        layers, bodies = [], []
        for name, value, base in zip(self.keys, arrays, self.view, strict=True):
            difference = torch.from_numpy(np.asarray(value, dtype=np.float32) - base.astype(np.float32))
            quantized = quantize_tensor(difference, bit_width=self.bit_width)
            unsigned = (quantized.values.numpy().reshape(-1) - qmin).astype(np.uint8)
            bodies.append(pack_codes(unsigned, self.bit_width))
            layers.append(
                {"name": name, "shape": list(value.shape), "scale": float(quantized.params.scale)}
            )
        header = json.dumps(
            {"round": int(round_index), "bit_width": self.bit_width, "layers": layers},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        body = b"".join(bodies)
        digest = hashlib.sha256(header + body).hexdigest().encode()
        blob = MAGIC + struct.pack("<I", len(header)) + header + body + digest
        # Update the view from the decoded bytes so server and clients match exactly.
        _, decoded = _decode(blob)
        self.view = _apply(self.view, self.keys, decoded)
        return [np.frombuffer(blob, dtype=np.uint8).copy()]


def decode_broadcast(
    parameters: Sequence[np.ndarray], view: list[np.ndarray] | None, keys: Sequence[str]
) -> list[np.ndarray]:
    """Client side: the model to train from, which is also the new view."""
    if not is_downlink_payload(parameters):
        return [np.array(item, copy=True) for item in parameters]
    if view is None:
        raise ValueError("received a quantized broadcast before any full model")
    _, layers = _decode(parameters[0].tobytes())
    return _apply(view, keys, layers)
