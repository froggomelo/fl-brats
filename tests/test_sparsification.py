import math

import numpy as np
import pytest
import torch

from fl_brats.federated.sparsification import (
    aggregate_sparse_updates,
    dequantized_sparse_values,
    create_sparse_update,
    deserialize_sparse_client_update,
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
