import torch

from fl_brats.federated.model_client import ModelFedAvgClient
from fl_brats.training.engine import proximal_penalty


def test_fedprox_mu_zero_adds_zero_penalty() -> None:
    model = torch.nn.Linear(2, 1)
    global_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    assert float(0.0 * proximal_penalty(model, global_parameters, torch.device("cpu"))) == 0.0


def test_fedprox_penalty_positive_when_parameters_move() -> None:
    model = torch.nn.Linear(2, 1)
    global_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    with torch.no_grad():
        model.weight.add_(1.0)

    assert proximal_penalty(model, global_parameters, torch.device("cpu")).item() > 0.0


def test_fedavg_client_does_not_enable_fedprox_loss() -> None:
    model = torch.nn.Linear(2, 1)
    client = ModelFedAvgClient(
        model,
        keys=["weight", "bias"],
        train_fn=lambda: {"train_loss": 0.0},
        evaluate_fn=lambda: {"loss": 0.0},
        examples=1,
        strategy="fedavg",
        fedprox_mu=10.0,
    )

    assert client.strategy == "fedavg"
    assert client._proximal_global_parameters is None
