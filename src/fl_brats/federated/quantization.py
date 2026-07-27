"""Packed, communication-time quantization primitives for federated updates.

The wire format deliberately contains integer codes plus JSON metadata; callers
must dequantize before doing FedAvg.  It is not a quantized-training format.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import torch

QuantizationScheme = Literal["symmetric", "asymmetric"]
QuantizationGranularity = Literal["per_tensor", "per_channel"]
RoundingMode = Literal["nearest", "stochastic"]
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class QuantizationParams:
    bit_width: int
    scale: torch.Tensor
    zero_point: torch.Tensor | None
    qmin: int
    qmax: int
    scheme: str
    granularity: str


@dataclass
class QuantizedTensor:
    values: torch.Tensor
    params: QuantizationParams
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype


@dataclass
class QuantizedLayerUpdate:
    name: str
    payload: QuantizedTensor
    num_parameters: int


@dataclass
class QuantizedClientUpdate:
    client_id: str
    round_index: int
    layers: list[QuantizedLayerUpdate]
    sample_count: int
    bit_width_assignment: dict[str, int]
    metadata_bytes: int = 0
    payload_bytes: int = 0
    global_model_version: int | None = None
    achieved_average_bit_width: float | None = None
    checksum: str = ""


@dataclass(frozen=True)
class LayerQuantizationStats:
    num_parameters: int
    importance: float = 1.0
    errors: Mapping[int, float] | None = None


@dataclass(frozen=True)
class AllocationResult:
    assignments: dict[str, int]
    achieved_average_bit_width: float
    estimated_total_error: float
    unused_bit_budget: float
    protected_layers: set[str]


def integer_range(bit_width: int, scheme: QuantizationScheme) -> tuple[int, int]:
    """Return representable signed symmetric or unsigned asymmetric code range."""
    if not isinstance(bit_width, int) or not 1 <= bit_width <= 16:
        raise ValueError("bit_width must be an integer from 1 through 16")
    return (
        (-(2 ** (bit_width - 1) - 1), 2 ** (bit_width - 1) - 1)
        if scheme == "symmetric"
        else (0, 2**bit_width - 1)
    )


def _validate_tensor(tensor: torch.Tensor) -> None:
    if tensor.is_complex() or not tensor.is_floating_point():
        raise TypeError("only real floating-point tensors can be quantized")


def _round(values: torch.Tensor, mode: RoundingMode, seed: int | None) -> torch.Tensor:
    if mode == "nearest":
        return torch.round(values)
    if seed is None:
        raise ValueError("stochastic rounding requires an explicit deterministic seed")
    generator = torch.Generator(device=values.device).manual_seed(seed)
    floor = torch.floor(values)
    return floor + (
        torch.rand(values.shape, device=values.device, generator=generator) < values - floor
    )


def quantize_tensor(
    tensor: torch.Tensor,
    *,
    bit_width: int,
    scheme: QuantizationScheme = "symmetric",
    granularity: QuantizationGranularity = "per_tensor",
    rounding: RoundingMode = "nearest",
    seed: int | None = None,
) -> QuantizedTensor:
    """Quantize a floating tensor without mutating it, retaining reconstruction data."""
    _validate_tensor(tensor)
    if scheme not in {"symmetric", "asymmetric"} or granularity not in {
        "per_tensor",
        "per_channel",
    }:
        raise ValueError("unsupported quantization scheme or granularity")
    if granularity == "per_channel" and tensor.ndim == 0:
        raise ValueError("per_channel quantization requires a tensor dimension")
    qmin, qmax = integer_range(bit_width, scheme)
    source = tensor.detach()
    reduce_dims = (
        tuple(range(1, source.ndim)) if granularity == "per_channel" else tuple(range(source.ndim))
    )
    keepdim = granularity == "per_channel"
    if scheme == "symmetric":
        maximum = (
            source.abs().amax(dim=reduce_dims, keepdim=keepdim) if reduce_dims else source.abs()
        )
        scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / qmax)
        zero_point = None
        codes = torch.clamp(_round(source / scale, rounding, seed), qmin, qmax)
    else:
        low = source.amin(dim=reduce_dims, keepdim=keepdim) if reduce_dims else source
        high = source.amax(dim=reduce_dims, keepdim=keepdim) if reduce_dims else source
        scale = torch.where(high == low, torch.ones_like(high), (high - low) / (qmax - qmin))
        zero_point = torch.clamp(
            _round(
                torch.tensor(qmin, device=source.device, dtype=source.dtype) - low / scale,
                "nearest",
                None,
            ),
            qmin,
            qmax,
        )
        codes = torch.clamp(_round(source / scale + zero_point, rounding, seed), qmin, qmax)
    return QuantizedTensor(
        codes.to(torch.int32),
        QuantizationParams(bit_width, scale, zero_point, qmin, qmax, scheme, granularity),
        tuple(source.shape),
        source.dtype,
    )


def dequantize_tensor(
    tensor: QuantizedTensor, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Reconstruct a quantized tensor on its payload device."""
    values = tensor.values.to(dtype)
    return (
        values - tensor.params.zero_point.to(dtype)
        if tensor.params.zero_point is not None
        else values
    ) * tensor.params.scale.to(dtype)


def quantization_error(original: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    error = original.detach().float() - reconstructed.detach().float()
    return {
        "mse": float(error.square().mean()),
        "relative_l2": float(error.norm() / original.detach().float().norm().clamp_min(1e-12)),
    }


def pack_quantized_values(values: torch.Tensor, bit_width: int) -> bytes:
    """Pack signed codes in little-endian, least-significant-code-first order."""
    qmin, qmax = integer_range(bit_width, "symmetric")
    flat = values.detach().to("cpu", torch.int64).reshape(-1)
    if bool(((flat < qmin) | (flat > qmax)).any()):
        raise ValueError("values are outside the signed quantization range")
    unsigned = (flat - qmin).tolist()
    if bit_width == 16:
        return struct.pack("<" + "H" * len(unsigned), *unsigned)
    output = bytearray(math.ceil(len(unsigned) * bit_width / 8))
    for index, value in enumerate(unsigned):
        position = index * bit_width
        output[position // 8] |= int(value) << (position % 8)
        if position % 8 + bit_width > 8:
            output[position // 8 + 1] |= int(value) >> (8 - position % 8)
    return bytes(output)


def unpack_quantized_values(
    payload: bytes, *, bit_width: int, count: int, device: torch.device
) -> torch.Tensor:
    """Invert :func:`pack_quantized_values`, discarding final-byte padding."""
    qmin, _ = integer_range(bit_width, "symmetric")
    expected = math.ceil(count * bit_width / 8)
    if len(payload) != expected:
        raise ValueError("packed payload length does not match value count")
    if bit_width == 16:
        return (
            torch.tensor(
                struct.unpack("<" + "H" * count, payload), device=device, dtype=torch.int32
            )
            + qmin
        )
    result: list[int] = []
    mask = (1 << bit_width) - 1
    for index in range(count):
        position = index * bit_width
        value = payload[position // 8] >> (position % 8)
        if position % 8 + bit_width > 8:
            value |= payload[position // 8 + 1] << (8 - position % 8)
        result.append((value & mask) + qmin)
    return torch.tensor(result, device=device, dtype=torch.int32)


def _tensor_metadata(name: str, item: QuantizedTensor) -> dict[str, Any]:
    params = item.params
    return {
        "name": name,
        "shape": item.original_shape,
        "dtype": str(item.original_dtype).replace("torch.", ""),
        "bit_width": params.bit_width,
        "qmin": params.qmin,
        "qmax": params.qmax,
        "scheme": params.scheme,
        "granularity": params.granularity,
        "scale": params.scale.detach().cpu().tolist(),
        "zero_point": None
        if params.zero_point is None
        else params.zero_point.detach().cpu().tolist(),
        "count": item.values.numel(),
    }


def serialize_client_update(update: QuantizedClientUpdate) -> bytes:
    """Serialize a packed update independently of Python object memory."""
    payloads = [
        pack_quantized_values(layer.payload.values, layer.payload.params.bit_width)
        for layer in update.layers
    ]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "client_id": update.client_id,
        "round_index": update.round_index,
        "global_model_version": update.global_model_version,
        "sample_count": update.sample_count,
        "bit_width_assignment": update.bit_width_assignment,
        "achieved_average_bit_width": update.achieved_average_bit_width,
        "layers": [_tensor_metadata(layer.name, layer.payload) for layer in update.layers],
        "lengths": [len(x) for x in payloads],
    }
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    body = b"".join(payloads)
    digest = hashlib.sha256(header + body).hexdigest().encode()
    packet = struct.pack("<I", len(header)) + header + body + digest
    update.metadata_bytes, update.payload_bytes, update.checksum = (
        4 + len(header) + len(digest),
        len(body),
        digest.decode(),
    )
    return packet


def deserialize_client_update(
    blob: bytes, *, device: torch.device = torch.device("cpu")
) -> QuantizedClientUpdate:
    """Validate checksum and reconstruct a typed packed client update."""
    if len(blob) < 4 + 64:
        raise ValueError("incomplete quantized update")
    header_len = struct.unpack("<I", blob[:4])[0]
    header, body, checksum = blob[4 : 4 + header_len], blob[4 + header_len : -64], blob[-64:]
    if hashlib.sha256(header + body).hexdigest().encode() != checksum:
        raise ValueError("quantized update checksum mismatch")
    meta = json.loads(header)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported quantized update schema")
    layers: list[QuantizedLayerUpdate] = []
    offset = 0
    for item, length in zip(meta["layers"], meta["lengths"], strict=True):
        raw = body[offset : offset + length]
        offset += length
        codes = unpack_quantized_values(
            raw, bit_width=item["bit_width"], count=item["count"], device=device
        ).reshape(item["shape"])
        scale = torch.tensor(item["scale"], device=device)
        zero = (
            None if item["zero_point"] is None else torch.tensor(item["zero_point"], device=device)
        )
        params = QuantizationParams(
            item["bit_width"],
            scale,
            zero,
            item["qmin"],
            item["qmax"],
            item["scheme"],
            item["granularity"],
        )
        layers.append(
            QuantizedLayerUpdate(
                item["name"],
                QuantizedTensor(codes, params, tuple(item["shape"]), getattr(torch, item["dtype"])),
                item["count"],
            )
        )
    if offset != len(body):
        raise ValueError("quantized update contains trailing bytes")
    return QuantizedClientUpdate(
        meta["client_id"],
        meta["round_index"],
        layers,
        meta["sample_count"],
        meta["bit_width_assignment"],
        4 + header_len + 64,
        len(body),
        meta.get("global_model_version"),
        meta.get("achieved_average_bit_width"),
        checksum.decode(),
    )


class BitWidthAllocator:
    """Deterministic greedy error-reduction-per-bit mixed-precision allocator."""

    def allocate(
        self,
        *,
        layer_statistics: Mapping[str, LayerQuantizationStats],
        candidate_bit_widths: Sequence[int],
        average_budget: float,
        protected_layers: set[str],
        protected_minimum: int = 8,
    ) -> AllocationResult:
        bits = sorted(set(candidate_bit_widths))
        if (
            not bits
            or len(bits) != len(candidate_bit_widths)
            or any(not 1 <= x <= 16 for x in bits)
        ):
            raise ValueError("candidate bit widths must be unique integers from 1 through 16")
        names = sorted(layer_statistics)
        if not names:
            return AllocationResult({}, 0.0, 0.0, average_budget, set())
        minimum: dict[str, int] = {}
        for name in names:
            value = next((b for b in bits if b >= protected_minimum), None)
            if name in protected_layers:
                if value is None:
                    raise ValueError("protected layer minimum is infeasible")
                minimum[name] = value
            else:
                minimum[name] = bits[0]
        total = sum(layer_statistics[n].num_parameters for n in names)
        used = sum(layer_statistics[n].num_parameters * int(minimum[n]) for n in names)
        limit = average_budget * total
        if used > limit + 1e-9:
            raise ValueError("average bit-width budget is infeasible")
        assigned = dict(minimum)

        def error(n: str, b: int) -> float:
            stat = layer_statistics[n]
            return float(
                stat.errors.get(b, stat.importance / (2 ** (2 * b)))
                if stat.errors
                else stat.importance / (2 ** (2 * b))
            )

        while True:
            choices = []
            for n in names:
                position = bits.index(assigned[n])
                if position + 1 == len(bits):
                    continue
                nxt = bits[position + 1]
                extra = layer_statistics[n].num_parameters * (nxt - assigned[n])
                if used + extra <= limit + 1e-9:
                    reduction = error(n, assigned[n]) - error(n, nxt)
                    choices.append((reduction / extra, reduction, n, nxt, extra))
            if not choices:
                break
            _, _, name, nxt, extra = sorted(choices, key=lambda x: (-x[0], -x[1], x[2]))[0]
            assigned[name] = nxt
            used += extra
        estimated = sum(error(n, assigned[n]) for n in names)
        return AllocationResult(
            assigned, used / total, estimated, (limit - used) / total, set(protected_layers)
        )


def eligible_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Only floating state entries participate in delta communication."""
    return {name: tensor for name, tensor in state.items() if tensor.is_floating_point()}


def state_delta(
    local: Mapping[str, torch.Tensor], global_state: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if set(local) != set(global_state):
        raise ValueError("state dictionaries have different keys")
    result: dict[str, torch.Tensor] = {}
    for name, value in local.items():
        if value.shape != global_state[name].shape:
            raise ValueError(f"state tensor shape mismatch for {name}")
        if value.is_floating_point():
            result[name] = value - global_state[name]
    return result


def quantize_update(
    *,
    client_id: str,
    round_index: int,
    local_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    sample_count: int,
    bit_width_assignment: Mapping[str, int],
    scheme: QuantizationScheme = "symmetric",
    granularity: QuantizationGranularity = "per_tensor",
    rounding: RoundingMode = "nearest",
    seed: int | None = None,
    residuals: Mapping[str, torch.Tensor] | None = None,
) -> tuple[QuantizedClientUpdate, dict[str, torch.Tensor]]:
    """Create an actual packed-delta candidate and local error-feedback residuals."""
    delta = state_delta(local_state, global_state)
    if set(delta) != set(bit_width_assignment):
        raise ValueError("bit-width assignment must exactly cover eligible update layers")
    layers: list[QuantizedLayerUpdate] = []
    next_residuals: dict[str, torch.Tensor] = {}
    for index, name in enumerate(sorted(delta)):
        corrected = delta[name]
        if residuals and name in residuals:
            if residuals[name].shape != corrected.shape:
                raise ValueError(f"error-feedback residual shape mismatch for {name}")
            corrected = corrected + residuals[name].to(corrected.device)
        payload = quantize_tensor(
            corrected,
            bit_width=bit_width_assignment[name],
            scheme=scheme,
            granularity=granularity,
            rounding=rounding,
            seed=None if seed is None else seed + index,
        )
        reconstructed = dequantize_tensor(payload, dtype=corrected.dtype)
        next_residuals[name] = (corrected - reconstructed).detach()
        layers.append(QuantizedLayerUpdate(name, payload, corrected.numel()))
    total = sum(layer.num_parameters for layer in layers)
    mean = sum(layer.num_parameters * layer.payload.params.bit_width for layer in layers) / total
    return (
        QuantizedClientUpdate(
            client_id=client_id,
            round_index=round_index,
            global_model_version=round_index,
            layers=layers,
            sample_count=sample_count,
            bit_width_assignment=dict(bit_width_assignment),
            achieved_average_bit_width=mean,
        ),
        next_residuals,
    )


def dequantized_delta(update: QuantizedClientUpdate) -> dict[str, torch.Tensor]:
    """Decode all layer payloads; callers must validate the expected round first."""
    return {layer.name: dequantize_tensor(layer.payload) for layer in update.layers}


def aggregate_deltas(
    global_state: Mapping[str, torch.Tensor],
    updates: Sequence[tuple[Mapping[str, torch.Tensor], int]],
) -> dict[str, torch.Tensor]:
    """Sample-weighted FP32 FedAvg delta aggregation."""
    total = sum(samples for _, samples in updates)
    if total <= 0:
        raise ValueError("at least one positive sample count is required")
    output = dict(global_state)
    for name, base in global_state.items():
        if not base.is_floating_point():
            continue
        values = [delta[name] for delta, _ in updates]
        if len(values) != len(updates):
            raise ValueError(f"missing delta for {name}")
        output[name] = base + sum(
            value.to(base.dtype) * samples / total
            for value, (_, samples) in zip(values, updates, strict=True)
        )
    return output
