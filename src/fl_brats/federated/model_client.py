"""Model-aware Flower client alias retaining standard FedAvg semantics."""

from .client import FedAvgClient


class ModelFedAvgClient(FedAvgClient):
    pass
