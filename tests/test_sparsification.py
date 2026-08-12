import math

import numpy as np
import pytest
import torch

from fl_brats.federated.sparsification import (
    aggregate_sparse_updates,
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
