import json
from types import SimpleNamespace

import numpy as np
import torch
from flwr.common import ndarrays_to_parameters

from fl_brats.federated.fedamm import aggregate_fedamm_arrays
from fl_brats.federated.parameters import state_to_arrays
from fl_brats.federated.research import (
    CheckpointManager,
    ClientRoundRecord,
    ExperimentState,
    RoundArtifactWriter,
)


def test_server_artifacts_and_checkpoint_are_atomic_and_named(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    writer = RoundArtifactWriter(tmp_path)
    state = ExperimentState(completed_round=1, uploaded_bytes=12)
    writer.write_client_records(
        [ClientRoundRecord(1, "client_0", 2, 0.3, 0.2, 0.6, 0.5, 0.4, 1.0, 12, 10, "completed")]
    )
    writer.write_round_record({"round": 1})
    writer.write_communication_record({"round": 1, "uploaded_bytes": 12})
    writer.write_state(state)
    manager = CheckpointManager(tmp_path, model, {"name": "tiny"})
    _, arrays = state_to_arrays(model)
    checkpoint = manager.save_round(1, ndarrays_to_parameters(arrays), state, {"loss": 0.2})

    assert checkpoint.exists()
    assert json.loads((tmp_path / "state.json").read_text())["completed_round"] == 1
    assert len((tmp_path / "metrics" / "clients.jsonl").read_text().splitlines()) == 1
    assert torch.load(checkpoint, weights_only=False)["model_config"] == {"name": "tiny"}


def test_fedamm_aggregates_modality_encoders_by_modality_counts() -> None:
    keys = ["modality_encoders.t1.weight", "decoder.weight"]
    first = SimpleNamespace(
        parameters=ndarrays_to_parameters([np.array([2.0]), np.array([10.0])]),
        num_examples=1,
        metrics={"fedamm_modality_count_t1": 10.0},
    )
    second = SimpleNamespace(
        parameters=ndarrays_to_parameters([np.array([8.0]), np.array([20.0])]),
        num_examples=3,
        metrics={"fedamm_modality_count_t1": 0.0},
    )
    arrays = aggregate_fedamm_arrays(keys, [(None, first), (None, second)])
    assert arrays[0].tolist() == [2.0]
    assert arrays[1].tolist() == [17.5]
