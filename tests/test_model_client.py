import torch
from torch.utils.data import DataLoader

from fl_brats.federated.model_client import ModelFedAvgClient, modality_count_metrics


def test_modality_count_metrics_stacks_tensor_masks() -> None:
    metrics = modality_count_metrics(
        [
            {"modality_mask": torch.tensor([True, False, True, False])},
            {"modality_mask": torch.tensor([True, True, False, False])},
        ]
    )

    assert metrics == {
        "fedamm_modality_count_t1": 2.0,
        "fedamm_modality_count_t1ce": 1.0,
        "fedamm_modality_count_t2": 1.0,
        "fedamm_modality_count_flair": 0.0,
    }


def test_client_uses_lightweight_metrics_unless_server_requests_expensive_metrics() -> None:
    model = torch.nn.Conv3d(4, 3, kernel_size=1)
    records = [
        {
            "image": torch.zeros(4, 2, 2, 2),
            "label": torch.zeros(3, 2, 2, 2),
        }
    ]
    loader = DataLoader(records, batch_size=1)
    client = ModelFedAvgClient(
        model,
        train_loader=loader,
        validation_loader=loader,
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        client_id="client_0",
    )
    parameters = client.get_model_parameters()

    _, _, lightweight = client.evaluate(parameters, {"compute_expensive_metrics": False})
    _, _, authoritative = client.evaluate(parameters, {"compute_expensive_metrics": True})

    assert "mean_dice" in lightweight
    assert "mean_hd95" not in lightweight
    assert "mean_sensitivity" not in lightweight
    assert "mean_hd95" in authoritative
    assert "mean_sensitivity" in authoritative


def test_client_state_round_trips_optimizer_across_resume(tmp_path) -> None:
    def make_client(state_dir, resume_path=None):
        model = torch.nn.Linear(2, 1)
        torch.manual_seed(0)
        optimizer = torch.optim.Adam(model.parameters())
        client = ModelFedAvgClient(
            model,
            train_fn=lambda: {"loss": 0.0},
            examples=1,
            optimizer=optimizer,
            client_id="client_0",
            state_dir=state_dir,
            resume_state_path=resume_path,
        )
        return client, model, optimizer

    first, model, optimizer = make_client(tmp_path / "a")
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    parameters = first.get_model_parameters()
    for server_round in range(1, 6):
        first.fit(parameters, {"server_round": server_round})
    saved = sorted(p.name for p in (tmp_path / "a").glob("*.pt"))
    assert saved == [f"client_0_round_000{r}.pt" for r in (3, 4, 5)]

    resumed, _, resumed_optimizer = make_client(
        tmp_path / "b", first.client_state_path(tmp_path / "a", 5)
    )
    resumed.fit(parameters, {"server_round": 6})
    expected = optimizer.state_dict()["state"][0]["exp_avg"]
    assert torch.equal(resumed_optimizer.state_dict()["state"][0]["exp_avg"], expected)
    assert (tmp_path / "b" / "client_0_round_0006.pt").is_file()
