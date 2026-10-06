"""Sparse Top-rho delta transmission for federated updates."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from .quantization import (
    QuantizationGranularity,
    QuantizationScheme,
    RoundingMode,
    QuantizedTensor,
    dequantize_tensor,
    pack_quantized_values,
    quantize_tensor,
    unpack_quantized_values,
)

SPARSE_UPDATE_MAGIC = b"FLBRATS_SPARSE_UPDATE_V1"
SCHEMA_VERSION = 1
# How the selected positions are sent: "flat" is one uint32 index per value,
# "csr" is compressed sparse row per layer (see encode_csr_indices).
INDEX_ENCODINGS = ("flat", "csr")

# Maps each layer with selected entries to (selected count, L2 norm of its
# selected values) and returns that layer's bit width.
BitWidthAllocation = Callable[[dict[str, tuple[int, float]]], dict[str, int]]


@dataclass
class LayerQuantizedValues:
    """Selected values quantized per layer: one segment per layer with entries,
    in index order, each with its own bit width and scale (mixed precision)."""

    segments: list[tuple[str, QuantizedTensor]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return sum(int(item.values.numel()) for _, item in self.segments)

    @property
    def average_bit_width(self) -> float:
        bits = sum(int(item.values.numel()) * item.params.bit_width for _, item in self.segments)
        return bits / max(self.count, 1)


@dataclass
class SparseClientUpdate:
    client_id: str
    round_index: int
    indices: torch.Tensor
    values: torch.Tensor | QuantizedTensor | LayerQuantizedValues
    total_parameters: int
    layers: list[dict[str, Any]]
    sample_count: int
    fraction: float
    metadata_bytes: int = 0
    index_bytes: int = 0
    value_bytes: int = 0
    global_model_version: int | None = None
    checksum: str = ""
    index_encoding: str = "flat"

    @property
    def selected_count(self) -> int:
        return int(self.indices.numel())

    @property
    def payload_bytes(self) -> int:
        return self.index_bytes + self.value_bytes


def is_sparse_parameter_payload(arrays: Sequence[np.ndarray]) -> bool:
    return (
        len(arrays) == 2
        and arrays[0].dtype == np.uint8
        and arrays[0].tobytes() == SPARSE_UPDATE_MAGIC
        and arrays[1].dtype == np.uint8
    )


def sparse_update_to_ndarrays(update: SparseClientUpdate) -> list[np.ndarray]:
    blob = serialize_sparse_client_update(update)
    return [
        np.frombuffer(SPARSE_UPDATE_MAGIC, dtype=np.uint8).copy(),
        np.frombuffer(blob, dtype=np.uint8).copy(),
    ]


def sparse_update_from_ndarrays(arrays: Sequence[np.ndarray]) -> SparseClientUpdate:
    if not is_sparse_parameter_payload(arrays):
        raise ValueError("Flower payload is not a sparse update packet")
    return deserialize_sparse_client_update(arrays[1].tobytes())


def _eligible_arrays(
    keys: Sequence[str], local: Sequence[np.ndarray], global_arrays: Sequence[np.ndarray]
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if len(keys) != len(local) or len(local) != len(global_arrays):
        raise ValueError("sparse update inputs have different tensor counts")
    result = []
    for name, local_array, global_array in zip(keys, local, global_arrays, strict=True):
        if local_array.shape != global_array.shape:
            raise ValueError(f"sparse update tensor shape mismatch for {name}")
        if np.issubdtype(local_array.dtype, np.floating):
            result.append((name, local_array, global_array))
    return result


def _layer_metadata(
    keys: Sequence[str], local: Sequence[np.ndarray], global_arrays: Sequence[np.ndarray]
) -> tuple[list[dict[str, Any]], int]:
    offset = 0
    layers: list[dict[str, Any]] = []
    for name, local_array, _ in _eligible_arrays(keys, local, global_arrays):
        count = int(local_array.size)
        layers.append(
            {
                "name": name,
                "shape": tuple(int(x) for x in local_array.shape),
                "dtype": str(local_array.dtype),
                "count": count,
                "offset": offset,
            }
        )
        offset += count
    return layers, offset


def create_sparse_update(
    *,
    client_id: str,
    round_index: int,
    keys: Sequence[str],
    local_arrays: Sequence[np.ndarray],
    global_arrays: Sequence[np.ndarray],
    sample_count: int,
    fraction: float,
    residual: torch.Tensor | None = None,
    quantize_values: bool = False,
    bit_width: int = 8,
    scheme: QuantizationScheme = "symmetric",
    granularity: QuantizationGranularity = "per_tensor",
    rounding: RoundingMode = "nearest",
    seed: int | None = None,
    allocate_bit_widths: BitWidthAllocation | None = None,
    index_encoding: str = "flat",
) -> tuple[SparseClientUpdate, torch.Tensor]:
    """Select global Top-rho delta entries and return the next residual vector.

    With ``allocate_bit_widths`` the selected values are quantized per layer at
    the bit width it assigns (mixed precision); otherwise at ``bit_width``.
    """
    if not 0 < fraction <= 1:
        raise ValueError("sparsification fraction must be in (0, 1]")
    if index_encoding not in INDEX_ENCODINGS:
        raise ValueError(f"sparse index encoding must be one of {INDEX_ENCODINGS}")
    layers, total = _layer_metadata(keys, local_arrays, global_arrays)
    if total <= 0:
        raise ValueError("sparse update contains no eligible floating tensors")
    pieces = [
        torch.from_numpy(
            # asarray: the difference of two 0-d arrays is a NumPy scalar.
            np.asarray(
                local.astype(np.float32, copy=False)
                - global_array.astype(np.float32, copy=False)
            )
        ).reshape(-1)
        for _, local, global_array in _eligible_arrays(keys, local_arrays, global_arrays)
    ]
    compensated = torch.cat(pieces)
    if residual is not None:
        if residual.numel() != compensated.numel():
            raise ValueError("sparsification residual length mismatch")
        compensated = compensated + residual.to(compensated)
    selected = int(math.ceil(fraction * total))
    if selected >= total:
        indices = torch.arange(total, dtype=torch.int64)
    else:
        _, indices = torch.topk(compensated.abs(), k=selected, largest=True, sorted=True)
        indices = indices.sort().values.to(torch.int64)
    selected_values = compensated[indices].contiguous()
    transmitted: torch.Tensor | QuantizedTensor | LayerQuantizedValues
    reconstructed_values: torch.Tensor
    if quantize_values and allocate_bit_widths is not None:
        transmitted = _quantize_per_layer(
            selected_values, indices, layers, allocate_bit_widths,
            scheme=scheme, granularity=granularity, rounding=rounding, seed=seed,
        )
        reconstructed_values = _dequantize_layers(transmitted, compensated.dtype)
    elif quantize_values:
        transmitted = quantize_tensor(
            selected_values,
            bit_width=bit_width,
            scheme=scheme,
            granularity=granularity,
            rounding=rounding,
            seed=seed,
        )
        reconstructed_values = dequantize_tensor(transmitted, dtype=compensated.dtype)
    else:
        transmitted = selected_values
        reconstructed_values = selected_values
    dense_reconstructed = torch.zeros_like(compensated)
    dense_reconstructed[indices] = reconstructed_values.to(dense_reconstructed)
    next_residual = (compensated - dense_reconstructed).detach()
    return (
        SparseClientUpdate(
            client_id=client_id,
            round_index=round_index,
            global_model_version=round_index,
            indices=indices,
            values=transmitted,
            total_parameters=total,
            layers=layers,
            sample_count=sample_count,
            fraction=fraction,
            index_encoding=index_encoding,
        ),
        next_residual,
    )


def _layer_slices(indices: torch.Tensor, layers: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    """(layer name, start, stop) of each layer's run in the sorted selected indices."""
    bounds = torch.tensor([int(layer["offset"]) for layer in layers] + [layers[-1]["offset"] + layers[-1]["count"]])
    positions = torch.searchsorted(indices, bounds).tolist()
    return [
        (str(layer["name"]), positions[i], positions[i + 1])
        for i, layer in enumerate(layers)
        if positions[i + 1] > positions[i]
    ]


def _quantize_per_layer(
    values: torch.Tensor,
    indices: torch.Tensor,
    layers: list[dict[str, Any]],
    allocate: BitWidthAllocation,
    *,
    scheme: QuantizationScheme,
    granularity: QuantizationGranularity,
    rounding: RoundingMode,
    seed: int | None,
) -> LayerQuantizedValues:
    if granularity != "per_tensor":
        raise ValueError("per-layer sparse quantization uses one scale per layer (per_tensor)")
    runs = _layer_slices(indices, layers)
    statistics = {name: (stop - start, float(values[start:stop].norm())) for name, start, stop in runs}
    widths = allocate(statistics)
    missing = sorted(set(statistics) - set(widths))
    if missing:
        raise ValueError(f"bit-width allocation is missing layers: {missing[:5]}")
    return LayerQuantizedValues(
        [
            (
                name,
                quantize_tensor(
                    values[start:stop], bit_width=int(widths[name]), scheme=scheme,
                    granularity="per_tensor", rounding=rounding, seed=seed,
                ),
            )
            for name, start, stop in runs
        ]
    )


def _dequantize_layers(values: LayerQuantizedValues, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if not values.segments:
        return torch.zeros(0, dtype=dtype)
    return torch.cat([dequantize_tensor(item, dtype=dtype) for _, item in values.segments])


def _index_dtype(total_parameters: int) -> tuple[str, np.dtype[Any]]:
    return (
        ("uint32", np.dtype("<u4"))
        if total_parameters < 2**32
        else ("uint64", np.dtype("<u8"))
    )


def _csr_layout(layer: dict[str, Any]) -> tuple[int, int]:
    """(rows, columns) of a layer seen as a matrix: one row per output channel."""
    shape, count = layer["shape"], int(layer["count"])
    rows = int(shape[0]) if len(shape) > 1 else 1
    return rows, count // rows


def _smallest_unsigned(limit: int) -> np.dtype[Any]:
    """Smallest little-endian unsigned dtype holding every value below ``limit``."""
    for name in ("<u1", "<u2", "<u4"):
        if limit <= int(np.iinfo(np.dtype(name)).max) + 1:
            return np.dtype(name)
    return np.dtype("<u8")


def encode_csr_indices(indices: np.ndarray, layers: list[dict[str, Any]]) -> tuple[bytes, list[int]]:
    """Sorted flat indices as compressed sparse row (CSR), layer by layer.

    Each layer with selected entries sends its row pointers (rows + 1 counts)
    and the column of every entry, both in the narrowest unsigned type that
    fits, so a conv weight with fewer than 65,536 columns needs 2 bytes per
    entry instead of 4. Layers with no entries send nothing. Returns the bytes
    and the entry count of every layer, which goes in the header.
    """
    ends = [int(layer["offset"]) + int(layer["count"]) for layer in layers]
    bounds = np.searchsorted(indices, [int(layers[0]["offset"]), *ends]) if layers else []
    parts: list[bytes] = []
    counts: list[int] = []
    for position, layer in enumerate(layers):
        local = indices[bounds[position] : bounds[position + 1]].astype(np.int64) - int(layer["offset"])
        counts.append(int(local.size))
        if not local.size:
            continue
        rows, columns = _csr_layout(layer)
        row, column = np.divmod(local, columns)
        row_pointers = np.searchsorted(row, np.arange(rows + 1))
        parts.append(row_pointers.astype(_smallest_unsigned(local.size + 1)).tobytes())
        parts.append(column.astype(_smallest_unsigned(columns)).tobytes())
    return b"".join(parts), counts


def decode_csr_indices(body: bytes, layers: list[dict[str, Any]], counts: Sequence[int]) -> np.ndarray:
    """Invert :func:`encode_csr_indices` to sorted flat indices."""
    if len(counts) != len(layers):
        raise ValueError("CSR entry counts do not match the layers")
    cursor, pieces = 0, []
    for layer, count in zip(layers, counts, strict=True):
        count = int(count)
        if count == 0:
            continue
        rows, columns = _csr_layout(layer)
        pointer_type, column_type = _smallest_unsigned(count + 1), _smallest_unsigned(columns)
        pointer_bytes, column_bytes = (rows + 1) * pointer_type.itemsize, count * column_type.itemsize
        if count < 0 or cursor + pointer_bytes + column_bytes > len(body):
            raise ValueError("sparse update CSR indices are truncated")
        row_pointers = np.frombuffer(body, pointer_type, rows + 1, cursor).astype(np.int64)
        cursor += pointer_bytes
        column = np.frombuffer(body, column_type, count, cursor).astype(np.int64)
        cursor += column_bytes
        if (
            row_pointers[0] != 0
            or row_pointers[-1] != count
            or bool((np.diff(row_pointers) < 0).any())
            or bool((column >= columns).any())
        ):
            raise ValueError("sparse update contains an invalid CSR layer")
        row = np.repeat(np.arange(rows, dtype=np.int64), np.diff(row_pointers))
        pieces.append(int(layer["offset"]) + row * columns + column)
    if cursor != len(body):
        raise ValueError("sparse update CSR indices have trailing bytes")
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)


def index_bytes_by_encoding(update: SparseClientUpdate) -> dict[str, int]:
    """Bytes the selected positions take in each index encoding."""
    indices = update.indices.detach().cpu().numpy()
    return {
        "flat": int(indices.size) * _index_dtype(update.total_parameters)[1].itemsize,
        "csr": len(encode_csr_indices(indices, update.layers)[0]),
    }


def _value_metadata(values: torch.Tensor | QuantizedTensor | LayerQuantizedValues) -> dict[str, Any]:
    if isinstance(values, LayerQuantizedValues):
        return {
            "encoding": "quantized_per_layer",
            "segments": [
                {
                    "layer": name,
                    "count": int(item.values.numel()),
                    "dtype": str(item.original_dtype).replace("torch.", ""),
                    "bit_width": item.params.bit_width,
                    "qmin": item.params.qmin,
                    "qmax": item.params.qmax,
                    "scheme": item.params.scheme,
                    "granularity": item.params.granularity,
                    "scale": item.params.scale.detach().cpu().tolist(),
                }
                for name, item in values.segments
            ],
        }
    if isinstance(values, QuantizedTensor):
        params = values.params
        return {
            "encoding": "quantized",
            "dtype": str(values.original_dtype).replace("torch.", ""),
            "bit_width": params.bit_width,
            "qmin": params.qmin,
            "qmax": params.qmax,
            "scheme": params.scheme,
            "granularity": params.granularity,
            "scale": params.scale.detach().cpu().tolist(),
            "zero_point": None
            if params.zero_point is None
            else params.zero_point.detach().cpu().tolist(),
        }
    return {"encoding": "float32", "dtype": "float32"}


def serialize_sparse_client_update(update: SparseClientUpdate) -> bytes:
    index_name, index_dtype = _index_dtype(update.total_parameters)
    csr_counts: list[int] | None = None
    if update.index_encoding == "csr":
        indices, csr_counts = encode_csr_indices(update.indices.detach().cpu().numpy(), update.layers)
    elif update.index_encoding == "flat":
        indices = update.indices.detach().cpu().numpy().astype(index_dtype, copy=False).tobytes()
    else:
        raise ValueError(f"sparse index encoding must be one of {INDEX_ENCODINGS}")
    if isinstance(update.values, LayerQuantizedValues):
        if any(item.params.scheme != "symmetric" for _, item in update.values.segments):
            raise ValueError("packed sparse quantized values require symmetric quantization")
        # Each layer's codes are packed separately, so every segment starts on a byte.
        values = b"".join(
            pack_quantized_values(item.values, item.params.bit_width) for _, item in update.values.segments
        )
    elif isinstance(update.values, QuantizedTensor):
        if update.values.params.scheme != "symmetric":
            raise ValueError("packed sparse quantized values require symmetric quantization")
        values = pack_quantized_values(update.values.values, update.values.params.bit_width)
    else:
        values = update.values.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "client_id": update.client_id,
        "round_index": update.round_index,
        "global_model_version": update.global_model_version,
        "sample_count": update.sample_count,
        "fraction": update.fraction,
        "selected_count": update.selected_count,
        "total_parameters": update.total_parameters,
        "index_dtype": index_name,
        "index_encoding": update.index_encoding,
        "layers": update.layers,
        "value": _value_metadata(update.values),
        "index_bytes": len(indices),
        "value_bytes": len(values),
    }
    if csr_counts is not None:
        metadata["csr_counts"] = csr_counts
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    body = indices + values
    digest = hashlib.sha256(header + body).hexdigest().encode()
    packet = struct.pack("<I", len(header)) + header + body + digest
    update.metadata_bytes = 4 + len(header) + len(digest)
    update.index_bytes = len(indices)
    update.value_bytes = len(values)
    update.checksum = digest.decode()
    return packet


def deserialize_sparse_client_update(
    blob: bytes, *, device: torch.device = torch.device("cpu")
) -> SparseClientUpdate:
    if len(blob) < 4 + 64:
        raise ValueError("incomplete sparse update")
    header_len = struct.unpack("<I", blob[:4])[0]
    header, body, checksum = blob[4 : 4 + header_len], blob[4 + header_len : -64], blob[-64:]
    if hashlib.sha256(header + body).hexdigest().encode() != checksum:
        raise ValueError("sparse update checksum mismatch")
    meta = json.loads(header)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported sparse update schema")
    count = int(meta["selected_count"])
    total = int(meta["total_parameters"])
    index_dtype = np.dtype("<u4") if meta["index_dtype"] == "uint32" else np.dtype("<u8")
    index_bytes = int(meta["index_bytes"])
    value_bytes = int(meta["value_bytes"])
    if len(body) != index_bytes + value_bytes:
        raise ValueError("sparse update body length mismatch")
    # Packets without "index_encoding" predate CSR and are flat.
    index_encoding = meta.get("index_encoding", "flat")
    if index_encoding == "csr":
        indices_np = decode_csr_indices(body[:index_bytes], list(meta["layers"]), meta["csr_counts"])
    elif index_encoding == "flat":
        if index_bytes != count * index_dtype.itemsize:
            raise ValueError("sparse update index length mismatch")
        indices_np = np.frombuffer(body[:index_bytes], dtype=index_dtype, count=count).astype(
            np.int64
        )
    else:
        raise ValueError("unsupported sparse index encoding")
    if (
        indices_np.size != count
        or bool((indices_np < 0).any())
        or bool((indices_np >= total).any())
    ):
        raise ValueError("sparse update contains invalid indices")
    if indices_np.size > 1 and bool((indices_np[1:] <= indices_np[:-1]).any()):
        raise ValueError("sparse update indices must be unique and sorted")
    raw_values = body[index_bytes:]
    value_meta = meta["value"]
    from .quantization import QuantizationParams

    if value_meta["encoding"] == "quantized_per_layer":
        segments: list[tuple[str, QuantizedTensor]] = []
        cursor = 0
        for segment in value_meta["segments"]:
            width, size = int(segment["bit_width"]), int(segment["count"])
            length = math.ceil(size * width / 8)
            codes = unpack_quantized_values(
                raw_values[cursor : cursor + length], bit_width=width, count=size, device=device
            )
            cursor += length
            segments.append(
                (
                    str(segment["layer"]),
                    QuantizedTensor(
                        codes.to(torch.int32),
                        QuantizationParams(
                            width,
                            torch.tensor(segment["scale"], device=device),
                            None,
                            int(segment["qmin"]),
                            int(segment["qmax"]),
                            str(segment["scheme"]),
                            str(segment["granularity"]),
                        ),
                        (size,),
                        getattr(torch, segment["dtype"]),
                    ),
                )
            )
        if cursor != len(raw_values) or sum(int(x["count"]) for x in value_meta["segments"]) != count:
            raise ValueError("per-layer sparse values do not match the selected count")
        values: torch.Tensor | QuantizedTensor | LayerQuantizedValues = LayerQuantizedValues(segments)
    elif value_meta["encoding"] == "quantized":
        codes = unpack_quantized_values(
            raw_values,
            bit_width=int(value_meta["bit_width"]),
            count=count,
            device=device,
        )
        scale = torch.tensor(value_meta["scale"], device=device)
        zero = (
            None
            if value_meta["zero_point"] is None
            else torch.tensor(value_meta["zero_point"], device=device)
        )
        values = QuantizedTensor(
            codes.to(torch.int32),
            QuantizationParams(
                int(value_meta["bit_width"]),
                scale,
                zero,
                int(value_meta["qmin"]),
                int(value_meta["qmax"]),
                str(value_meta["scheme"]),
                str(value_meta["granularity"]),
            ),
            (count,),
            getattr(torch, value_meta["dtype"]),
        )
    elif value_meta["encoding"] == "float32":
        values = torch.from_numpy(
            np.frombuffer(raw_values, dtype=np.dtype("<f4"), count=count).copy()
        ).to(device)
    else:
        raise ValueError("unsupported sparse value encoding")
    return SparseClientUpdate(
        client_id=meta["client_id"],
        round_index=int(meta["round_index"]),
        global_model_version=meta.get("global_model_version"),
        indices=torch.from_numpy(indices_np.copy()).to(device),
        values=values,
        total_parameters=total,
        layers=list(meta["layers"]),
        sample_count=int(meta["sample_count"]),
        fraction=float(meta["fraction"]),
        metadata_bytes=4 + header_len + 64,
        index_bytes=index_bytes,
        value_bytes=value_bytes,
        checksum=checksum.decode(),
        index_encoding=index_encoding,
    )


def dequantized_sparse_values(update: SparseClientUpdate) -> torch.Tensor:
    if isinstance(update.values, LayerQuantizedValues):
        return _dequantize_layers(update.values)
    if isinstance(update.values, QuantizedTensor):
        return dequantize_tensor(update.values)
    return update.values


def sparse_delta_to_arrays(
    update: SparseClientUpdate, keys: Sequence[str], reference_arrays: Sequence[np.ndarray]
) -> list[np.ndarray]:
    """Reconstruct full model arrays by applying sparse delta to floating references."""
    output = [array.copy() for array in reference_arrays]
    values = dequantized_sparse_values(update).detach().cpu()
    cursor = 0
    for layer in update.layers:
        name = str(layer["name"])
        if name not in keys:
            raise ValueError(f"sparse update references unknown layer {name}")
        position = keys.index(name)
        base = torch.from_numpy(
            reference_arrays[position].astype(np.float32, copy=False)
        ).reshape(-1)
        count = int(layer["count"])
        offset = int(layer["offset"])
        mask = (update.indices.cpu() >= offset) & (update.indices.cpu() < offset + count)
        local_indices = (update.indices.cpu()[mask] - offset).long()
        delta = torch.zeros(count, dtype=torch.float32)
        take = int(mask.sum())
        if take:
            delta[local_indices] = values[cursor : cursor + take].to(delta)
        cursor += take
        output[position] = (
            (base + delta)
            .reshape(reference_arrays[position].shape)
            .numpy()
            .astype(reference_arrays[position].dtype, copy=False)
        )
    if cursor != update.selected_count:
        raise ValueError("sparse update selected values were not fully consumed")
    return output


def aggregate_sparse_updates(
    updates: Sequence[tuple[SparseClientUpdate, int]],
    keys: Sequence[str],
    reference_arrays: Sequence[np.ndarray],
) -> list[np.ndarray]:
    total_examples = sum(examples for _, examples in updates)
    if total_examples <= 0:
        raise ValueError("at least one positive sparse update sample count is required")
    output = [array.copy() for array in reference_arrays]
    accumulators: dict[str, torch.Tensor] = {}
    for update, examples in updates:
        values = dequantized_sparse_values(update).detach().cpu()
        cursor = 0
        for layer in update.layers:
            name = str(layer["name"])
            count = int(layer["count"])
            offset = int(layer["offset"])
            mask = (update.indices.cpu() >= offset) & (update.indices.cpu() < offset + count)
            local_indices = (update.indices.cpu()[mask] - offset).long()
            if name not in accumulators:
                accumulators[name] = torch.zeros(count, dtype=torch.float32)
            take = int(mask.sum())
            if take:
                accumulators[name][local_indices] += (
                    values[cursor : cursor + take].to(torch.float32) * examples / total_examples
                )
            cursor += take
        if cursor != update.selected_count:
            raise ValueError("sparse update selected values were not fully consumed")
    for name, delta in accumulators.items():
        position = keys.index(name)
        base = torch.from_numpy(
            reference_arrays[position].astype(np.float32, copy=False)
        ).reshape(-1)
        output[position] = (
            (base + delta)
            .reshape(reference_arrays[position].shape)
            .numpy()
            .astype(reference_arrays[position].dtype, copy=False)
        )
    return output
