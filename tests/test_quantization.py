import math

import pytest
import torch

from fl_brats.federated.communication import quantized_payload_metrics
from fl_brats.federated.quantization import (
    BitWidthAllocator,
    LayerQuantizationStats,
    QuantizedClientUpdate,
    QuantizedLayerUpdate,
    aggregate_deltas,
    dequantize_tensor,
    dequantized_delta,
    deserialize_client_update,
    pack_quantized_values,
    quantize_tensor,
    quantize_update,
    serialize_client_update,
    state_delta,
    unpack_quantized_values,
)


@pytest.mark.parametrize("bits", [2, 4, 8, 16])
def test_packing_round_trip_including_padding(bits: int) -> None:
    qmin, qmax = -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
    values = torch.tensor([qmin, -1, 0, 1, qmax], dtype=torch.int32)
    payload = pack_quantized_values(values, bits)
    assert len(payload) == math.ceil(values.numel() * bits / 8)
    assert torch.equal(
        unpack_quantized_values(payload, bit_width=bits, count=5, device=torch.device("cpu")),
        values,
    )


def test_quantization_zero_asymmetric_and_stochastic() -> None:
    zero = quantize_tensor(torch.zeros(5), bit_width=4)
    assert torch.isfinite(dequantize_tensor(zero)).all()
    assert torch.equal(zero.values, torch.zeros(5, dtype=torch.int32))
    asymmetric = quantize_tensor(torch.tensor([-2.0, 0.5, 5.0]), bit_width=8, scheme="asymmetric")
    assert torch.allclose(dequantize_tensor(asymmetric), torch.tensor([-2.0, 0.5, 5.0]), atol=0.04)
    first = quantize_tensor(torch.tensor([0.1, 0.2]), bit_width=2, rounding="stochastic", seed=7)
    second = quantize_tensor(torch.tensor([0.1, 0.2]), bit_width=2, rounding="stochastic", seed=7)
    assert torch.equal(first.values, second.values)


def test_wire_round_trip_and_real_accounting() -> None:
    payload = quantize_tensor(torch.tensor([-1.0, 0.0, 1.0]), bit_width=4)
    update = QuantizedClientUpdate(
        "client_0",
        3,
        [QuantizedLayerUpdate("weight", payload, 3)],
        4,
        {"weight": 4},
        global_model_version=3,
    )
    wire = serialize_client_update(update)
    restored = deserialize_client_update(wire)
    assert torch.allclose(dequantize_tensor(restored.layers[0].payload), dequantize_tensor(payload))
    metrics = quantized_payload_metrics(update)
    assert metrics["total_serialized_bytes"] == len(wire)
    assert metrics["metadata_bytes"] + metrics["actual_packed_payload_bytes"] == len(wire)


def test_allocator_budget_protection_and_delta_fedavg() -> None:
    allocator = BitWidthAllocator()
    result = allocator.allocate(
        layer_statistics={
            "body": LayerQuantizationStats(100, 10),
            "output": LayerQuantizationStats(10, 1),
        },
        candidate_bit_widths=[2, 4, 8],
        average_budget=4,
        protected_layers={"output"},
        protected_minimum=8,
    )
    assert result.achieved_average_bit_width <= 4
    assert result.assignments["output"] == 8
    global_state = {"weight": torch.tensor([1.0]), "counter": torch.tensor(1)}
    first = {"weight": torch.tensor([2.0]), "counter": torch.tensor(2)}
    second = {"weight": torch.tensor([4.0]), "counter": torch.tensor(3)}
    output = aggregate_deltas(
        global_state,
        [(state_delta(first, global_state), 1), (state_delta(second, global_state), 3)],
    )
    assert torch.allclose(output["weight"], torch.tensor([3.5]))
    assert output["counter"] == 1


def test_quantized_update_is_delta_and_error_feedback_is_local() -> None:
    global_state = {"weight": torch.tensor([1.0, 2.0]), "counter": torch.tensor(0)}
    local_state = {"weight": torch.tensor([2.0, 0.0]), "counter": torch.tensor(1)}
    update, residuals = quantize_update(
        client_id="client_0",
        round_index=2,
        local_state=local_state,
        global_state=global_state,
        sample_count=5,
        bit_width_assignment={"weight": 2},
    )
    assert update.global_model_version == 2
    assert set(residuals) == {"weight"}
    assert set(dequantized_delta(update)) == {"weight"}


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        quantize_tensor(torch.ones(1), bit_width=0)
    with pytest.raises(TypeError):
        quantize_tensor(torch.ones(1, dtype=torch.int64), bit_width=4)
