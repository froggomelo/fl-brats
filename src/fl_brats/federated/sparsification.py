"""Sparse Top-rho delta transmission for federated updates."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
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


@dataclass
class SparseClientUpdate:
    client_id: str
    round_index: int
    indices: torch.Tensor
    values: torch.Tensor | QuantizedTensor
    total_parameters: int
    layers: list[dict[str, Any]]
    sample_count: int
    fraction: float
    metadata_bytes: int = 0
    index_bytes: int = 0
    value_bytes: int = 0
    global_model_version: int | None = None
    checksum: str = ""

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
) -> tuple[SparseClientUpdate, torch.Tensor]:
    """Select global Top-rho delta entries and return the next residual vector."""
    if not 0 < fraction <= 1:
        raise ValueError("sparsification fraction must be in (0, 1]")
    layers, total = _layer_metadata(keys, local_arrays, global_arrays)
    if total <= 0:
        raise ValueError("sparse update contains no eligible floating tensors")
    pieces = [
        torch.from_numpy(
            local.astype(np.float32, copy=False)
            - global_array.astype(np.float32, copy=False)
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
    transmitted: torch.Tensor | QuantizedTensor
    reconstructed_values: torch.Tensor
    if quantize_values:
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
        ),
        next_residual,
    )


def _index_dtype(total_parameters: int) -> tuple[str, np.dtype[Any]]:
    return (
        ("uint32", np.dtype("<u4"))
        if total_parameters < 2**32
        else ("uint64", np.dtype("<u8"))
    )


def _value_metadata(values: torch.Tensor | QuantizedTensor) -> dict[str, Any]:
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
    indices = update.indices.detach().cpu().numpy().astype(index_dtype, copy=False).tobytes()
    if isinstance(update.values, QuantizedTensor):
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
        "layers": update.layers,
        "value": _value_metadata(update.values),
        "index_bytes": len(indices),
        "value_bytes": len(values),
    }
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
    indices_np = np.frombuffer(body[:index_bytes], dtype=index_dtype, count=count).astype(
        np.int64
    )
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
    if value_meta["encoding"] == "quantized":
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
        from .quantization import QuantizationParams

        values: torch.Tensor | QuantizedTensor = QuantizedTensor(
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
    )


def dequantized_sparse_values(update: SparseClientUpdate) -> torch.Tensor:
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
