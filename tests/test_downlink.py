from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays

from fl_brats.federated.downlink import (
    DownlinkEncoder,
    decode_broadcast,
    is_downlink_payload,
    pack_codes,
    unpack_codes,
)
from fl_brats.federated.model_client import ModelFedAvgClient
from fl_brats.federated.parameters import state_to_arrays
from fl_brats.federated.strategy import create_research_fedavg, downlink_bit_width

KEYS = ["conv.weight", "conv.bias"]


def _model(rng: np.random.Generator) -> list[np.ndarray]:
    return [rng.normal(size=(4, 3, 2)).astype(np.float32), rng.normal(size=(4,)).astype(np.float32)]


@pytest.mark.parametrize("bit_width", [2, 4, 8])
def test_pack_codes_round_trips(bit_width: int) -> None:
    codes = np.random.default_rng(0).integers(0, 2**bit_width, size=37).astype(np.uint8)
    packed = pack_codes(codes, bit_width)
    assert len(packed) == -(-37 // (8 // bit_width))
    np.testing.assert_array_equal(unpack_codes(packed, bit_width, 37), codes)


def test_first_broadcast_is_full_and_later_ones_are_quantized_and_smaller() -> None:
    rng = np.random.default_rng(1)
    encoder = DownlinkEncoder(KEYS, 8)
    model = _model(rng)
    first = encoder.encode(model, 1)
    assert not is_downlink_payload(first)
    view = decode_broadcast(first, None, KEYS)
    model = [item + 0.01 * rng.normal(size=item.shape).astype(np.float32) for item in model]
    second = encoder.encode(model, 2)
    assert is_downlink_payload(second)
    # Header and checksum dominate this toy model; the codes are one byte per value.
    assert second[0].nbytes < sum(item.nbytes for item in model) + 400
    view = decode_broadcast(second, view, KEYS)
    for client, server in zip(view, encoder.view):
        np.testing.assert_array_equal(client, server)  # bit-exact shared view
        assert client.dtype == np.float32


@pytest.mark.parametrize("bit_width", [2, 8])
def test_error_feedback_drives_the_view_to_a_fixed_model(bit_width: int) -> None:
    rng = np.random.default_rng(2)
    encoder = DownlinkEncoder(KEYS, bit_width)
    model = _model(rng)
    view = decode_broadcast(encoder.encode([item * 0 for item in model], 1), None, KEYS)
    errors = []
    for round_index in range(2, 12):
        view = decode_broadcast(encoder.encode(model, round_index), view, KEYS)
        errors.append(max(float(np.abs(v - m).max()) for v, m in zip(view, model)))
    # The residual is resent every round, so the error never grows; at 8 bits each
    # round divides it by about 2**7.
    assert all(later <= earlier for earlier, later in zip(errors, errors[1:]))
    if bit_width == 8:
        assert errors[2] < 1e-5 * max(float(np.abs(item).max()) for item in model)


def test_quantized_broadcast_without_a_view_or_with_bad_bytes_is_rejected() -> None:
    rng = np.random.default_rng(3)
    encoder = DownlinkEncoder(KEYS, 8)
    model = _model(rng)
    view = decode_broadcast(encoder.encode(model, 1), None, KEYS)
    payload = encoder.encode([item + 1 for item in model], 2)
    with pytest.raises(ValueError, match="before any full model"):
        decode_broadcast(payload, None, KEYS)
    corrupt = payload[0].copy()
    corrupt[-80] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        decode_broadcast([corrupt], view, KEYS)
    with pytest.raises(ValueError, match="bit width"):
        DownlinkEncoder(KEYS, 3)


def test_downlink_bit_width_reads_the_communication_config() -> None:
    assert downlink_bit_width({}) is None
    assert downlink_bit_width({"quantization": {"communication": {"quantize_downlink": False}}}) is None
    assert downlink_bit_width({"quantization": {"communication": {"quantize_downlink": True}}}) == 8
    config = {"quantization": {"communication": {"quantize_downlink": True, "downlink_bit_width": 4}}}
    assert downlink_bit_width(config) == 4


def test_strategy_broadcasts_one_payload_and_rebases_dense_averages(tmp_path) -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    strategy = create_research_fedavg(
        {"num_clients": 2, "quantization": {"communication": {"quantize_downlink": True}}},
        run_dir=tmp_path,
        model=model,
    )
    keys, arrays = state_to_arrays(model)
    proxies = [SimpleNamespace(cid="0"), SimpleNamespace(cid="1")]
    manager = SimpleNamespace(
        num_available=lambda: 2,
        sample=lambda num_clients, min_num_clients=None, criterion=None: proxies,
        wait_for=lambda *args, **kwargs: True,
    )
    first = strategy.configure_fit(1, ndarrays_to_parameters(arrays), manager)
    assert len(first) == 2 and all(isinstance(ins, FitIns) for _, ins in first)
    assert not is_downlink_payload(parameters_to_ndarrays(first[0][1].parameters))

    moved = [item + np.float32(0.0123) for item in arrays]
    strategy._current_arrays = moved
    second = strategy.configure_fit(2, ndarrays_to_parameters(moved), manager)
    payloads = [parameters_to_ndarrays(ins.parameters) for _, ins in second]
    assert all(is_downlink_payload(payload) for payload in payloads)
    np.testing.assert_array_equal(payloads[0][0], payloads[1][0])

    # A dense client average equal to the view means "no update": the FP32 model stays.
    rebased = parameters_to_ndarrays(
        strategy.rebase_dense_aggregate(ndarrays_to_parameters(strategy.downlink.view))
    )
    for result, expected in zip(rebased, moved):
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)


def test_client_trains_from_the_view_and_reports_received_bytes() -> None:
    model = torch.nn.Linear(3, 2)
    client = ModelFedAvgClient(
        model,
        train_fn=lambda: {"loss": 0.0},
        examples=1,
        client_id="client_0",
        quantization_config={"communication": {"quantize_downlink": True}},
    )
    keys, arrays = state_to_arrays(model)
    encoder = DownlinkEncoder(keys, 8)
    full = encoder.encode(arrays, 1)
    _, _, metrics = client.fit(full, {"server_round": 1})
    assert metrics["download_payload_bytes"] == sum(item.nbytes for item in arrays)

    payload = encoder.encode([item + np.float32(0.5) for item in arrays], 2)
    uploaded, _, metrics = client.fit(payload, {"server_round": 2})
    assert metrics["download_payload_bytes"] == payload[0].nbytes
    for local, view in zip(uploaded, encoder.view):
        np.testing.assert_array_equal(local, view)  # train_fn is a no-op
