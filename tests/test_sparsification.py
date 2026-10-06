import math

import numpy as np
import pytest
import torch

from fl_brats.federated.sparsification import (
    aggregate_sparse_updates,
    dequantized_sparse_values,
    create_sparse_update,
    decode_csr_indices,
    deserialize_sparse_client_update,
    encode_csr_indices,
    index_bytes_by_encoding,
    is_sparse_parameter_payload,
    serialize_sparse_client_update,
    sparse_update_to_ndarrays,
)


def test_toprho_sparse_wire_round_trip_and_residual() -> None:
    keys = ["a", "b"]
    global_arrays = [np.zeros(4, dtype=np.float32), np.zeros(2, dtype=np.float32)]
    local_arrays = [
        np.array([1.0, -5.0, 2.0, 0.5], dtype=np.float32),
        np.array([3.0, -4.0], dtype=np.float32),
    ]

    update, residual = create_sparse_update(
        client_id="client_0",
        round_index=1,
        keys=keys,
        local_arrays=local_arrays,
        global_arrays=global_arrays,
        sample_count=3,
        fraction=0.5,
    )
    blob = serialize_sparse_client_update(update)
    restored = deserialize_sparse_client_update(blob)

    assert restored.selected_count == math.ceil(0.5 * 6)
    assert restored.indices.tolist() == [1, 4, 5]
    assert torch.allclose(restored.values, torch.tensor([-5.0, 3.0, -4.0]))
    assert torch.allclose(residual, torch.tensor([1.0, 0.0, 2.0, 0.5, 0.0, 0.0]))
    assert restored.payload_bytes <= sum(array.nbytes for array in local_arrays)


def test_sparse_payload_uses_uint8_flower_arrays() -> None:
    update, _ = create_sparse_update(
        client_id="client_0",
        round_index=1,
        keys=["weight"],
        local_arrays=[np.array([1.0, 0.0], dtype=np.float32)],
        global_arrays=[np.zeros(2, dtype=np.float32)],
        sample_count=1,
        fraction=0.5,
    )

    arrays = sparse_update_to_ndarrays(update)

    assert is_sparse_parameter_payload(arrays)
    assert arrays[0].dtype == np.uint8
    assert arrays[1].dtype == np.uint8


def test_sparse_updates_are_sample_weighted_deltas() -> None:
    keys = ["weight"]
    base = [np.array([10.0, 10.0, 10.0], dtype=np.float32)]
    first, _ = create_sparse_update(
        client_id="client_0",
        round_index=1,
        keys=keys,
        local_arrays=[np.array([12.0, 10.0, 10.0], dtype=np.float32)],
        global_arrays=base,
        sample_count=1,
        fraction=1.0,
    )
    second, _ = create_sparse_update(
        client_id="client_1",
        round_index=1,
        keys=keys,
        local_arrays=[np.array([10.0, 16.0, 10.0], dtype=np.float32)],
        global_arrays=base,
        sample_count=3,
        fraction=1.0,
    )

    output = aggregate_sparse_updates([(first, 1), (second, 3)], keys, base)

    assert output[0].tolist() == pytest.approx([10.5, 14.5, 10.0])


def test_sparse_quantized_values_are_bitpacked() -> None:
    update, _ = create_sparse_update(
        client_id="client_0",
        round_index=1,
        keys=["weight"],
        local_arrays=[np.array([-1.0, 0.0, 1.0, 0.5], dtype=np.float32)],
        global_arrays=[np.zeros(4, dtype=np.float32)],
        sample_count=1,
        fraction=1.0,
        quantize_values=True,
        bit_width=4,
    )

    restored = deserialize_sparse_client_update(serialize_sparse_client_update(update))

    assert restored.value_bytes == 2
    assert torch.allclose(restored.values.values.float(), update.values.values.float())


def test_invalid_fraction_is_rejected() -> None:
    with pytest.raises(ValueError, match="fraction"):
        create_sparse_update(
            client_id="client_0",
            round_index=1,
            keys=["weight"],
            local_arrays=[np.ones(1, dtype=np.float32)],
            global_arrays=[np.zeros(1, dtype=np.float32)],
            sample_count=1,
            fraction=0.0,
        )


def test_sparse_values_are_quantized_per_layer_at_allocated_bit_widths() -> None:
    torch.manual_seed(0)
    keys = ["encoder.weight", "head.weight", "unselected.weight"]
    global_arrays = [np.zeros(40, dtype=np.float32), np.zeros(10, dtype=np.float32), np.zeros(5, dtype=np.float32)]
    local_arrays = [
        np.random.default_rng(0).normal(size=40).astype(np.float32),
        np.random.default_rng(1).normal(size=10).astype(np.float32) * 3,
        np.full(5, 1e-6, dtype=np.float32),
    ]
    seen: dict[str, tuple[int, float]] = {}

    def allocate(statistics: dict[str, tuple[int, float]]) -> dict[str, int]:
        seen.update(statistics)
        return {name: 8 if name.startswith("head") else 2 for name in statistics}

    update, residual = create_sparse_update(
        client_id="client_0", round_index=3, keys=keys, local_arrays=local_arrays,
        global_arrays=global_arrays, sample_count=5, fraction=0.5,
        quantize_values=True, allocate_bit_widths=allocate,
    )
    # Only layers with selected entries are allocated and transmitted.
    assert set(seen) == {"encoder.weight", "head.weight"}
    assert sum(count for count, _ in seen.values()) == update.selected_count == 28
    blob = serialize_sparse_client_update(update)
    restored = deserialize_sparse_client_update(blob)
    widths = {name: item.params.bit_width for name, item in restored.values.segments}
    assert widths == {"encoder.weight": 2, "head.weight": 8}
    counts = {name: item.values.numel() for name, item in restored.values.segments}
    assert restored.value_bytes == sum(math.ceil(counts[n] * widths[n] / 8) for n in counts)
    # Error feedback: the residual is exactly what the server cannot reconstruct.
    dense = np.concatenate(local_arrays)
    reconstructed = torch.zeros(dense.size)
    reconstructed[restored.indices] = dequantized_sparse_values(restored)
    assert torch.allclose(reconstructed + residual, torch.from_numpy(dense), atol=1e-6)
    # Aggregation applies each layer's dequantized values to that layer only.
    merged = aggregate_sparse_updates([(restored, 5)], keys, global_arrays)
    for position in range(3):
        start = sum(a.size for a in local_arrays[:position])
        expected = reconstructed[start : start + local_arrays[position].size].numpy()
        assert np.allclose(merged[position], expected, atol=1e-6)


def test_client_mixed_allocation_respects_budget_and_protection() -> None:
    from fl_brats.federated.model_client import ModelFedAvgClient

    client = ModelFedAvgClient.__new__(ModelFedAvgClient)
    client.client_id = "client_2"
    client.quantization_config = {
        "mixed": {
            "candidate_bit_widths": [2, 4, 8],
            "client_average_bit_width_budgets": {"client_2": 4.0},
            "protected_layer_patterns": ["output"],
            "protected_layer_minimum_bit_width": 8,
        }
    }
    allocate = client._sparse_bit_width_allocation()
    widths = allocate({"model.output.weight": (10, 1.0), "model.conv.weight": (90, 5.0), "model.small.weight": (20, 0.1)})
    assert widths["model.output.weight"] == 8
    counts = {"model.output.weight": 10, "model.conv.weight": 90, "model.small.weight": 20}
    average = sum(counts[n] * widths[n] for n in counts) / sum(counts.values())
    assert average <= 4.0 + 1e-9
    assert set(widths.values()) <= {2, 4, 8}


def _conv_like_update(fraction: float, index_encoding: str, **kwargs):
    rng = np.random.default_rng(4)
    keys = ["conv.weight", "conv.bias", "norm.scale", "empty.weight"]
    shapes = [(8, 4, 3, 3, 3), (8,), (), (2, 3)]
    global_arrays = [np.zeros(shape, dtype=np.float32) for shape in shapes]
    local_arrays = [np.asarray(rng.normal(size=shape), dtype=np.float32) for shape in shapes[:3]]
    local_arrays.append(np.zeros(shapes[3], dtype=np.float32))  # never selected
    return keys, global_arrays, create_sparse_update(
        client_id="client_0", round_index=2, keys=keys, local_arrays=local_arrays,
        global_arrays=global_arrays, sample_count=4, fraction=fraction,
        index_encoding=index_encoding, **kwargs,
    )[0]


@pytest.mark.parametrize("quantize", [False, True])
def test_csr_indices_round_trip_and_match_the_flat_encoding(quantize: bool) -> None:
    options = {"quantize_values": True, "bit_width": 4} if quantize else {}
    keys, global_arrays, flat = _conv_like_update(0.3, "flat", **options)
    _, _, csr = _conv_like_update(0.3, "csr", **options)
    restored_flat = deserialize_sparse_client_update(serialize_sparse_client_update(flat))
    restored_csr = deserialize_sparse_client_update(serialize_sparse_client_update(csr))

    assert restored_csr.index_encoding == "csr"
    assert torch.equal(restored_csr.indices, restored_flat.indices)
    assert torch.equal(dequantized_sparse_values(restored_csr), dequantized_sparse_values(restored_flat))
    assert restored_csr.value_bytes == restored_flat.value_bytes
    assert restored_csr.index_bytes < restored_flat.index_bytes
    assert index_bytes_by_encoding(restored_csr) == {
        "flat": restored_flat.index_bytes,
        "csr": restored_csr.index_bytes,
    }
    for a, b in zip(
        aggregate_sparse_updates([(restored_csr, 4)], keys, global_arrays),
        aggregate_sparse_updates([(restored_flat, 4)], keys, global_arrays),
    ):
        np.testing.assert_array_equal(a, b)


def test_csr_layout_is_row_pointers_then_narrow_columns() -> None:
    layers = [{"name": "w", "shape": (3, 2, 2), "count": 12, "offset": 5}]
    # Entries (row, column): (0, 1), (0, 3), (2, 0), so row 1 is empty.
    body, counts = encode_csr_indices(np.array([6, 8, 13]), layers)
    assert counts == [3]
    assert body == bytes([0, 2, 2, 3]) + bytes([1, 3, 0])  # uint8 pointers and columns
    assert decode_csr_indices(body, layers, counts).tolist() == [6, 8, 13]


def test_corrupt_csr_indices_are_rejected() -> None:
    layers = [{"name": "w", "shape": (3, 2, 2), "count": 12, "offset": 0}]
    body, counts = encode_csr_indices(np.array([1, 3, 8]), layers)
    with pytest.raises(ValueError, match="invalid CSR"):
        decode_csr_indices(bytes([0, 2, 2, 3]) + bytes([1, 4, 0]), layers, counts)  # column 4 >= 4
    with pytest.raises(ValueError, match="invalid CSR"):
        decode_csr_indices(bytes([0, 2, 1, 3]) + body[4:], layers, counts)  # pointers decrease
    with pytest.raises(ValueError, match="truncated"):
        decode_csr_indices(body[:-1], layers, counts)
    with pytest.raises(ValueError, match="trailing"):
        decode_csr_indices(body + b"\0", layers, counts)
    with pytest.raises(ValueError, match="index encoding"):
        _conv_like_update(0.3, "bitmap")
