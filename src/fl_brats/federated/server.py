"""Flower server startup isolated from experiment orchestration."""

from typing import Any

from .strategy import create_fedavg


def start_server(address: str, config: dict[str, Any], strategy: Any = None) -> None:
    import flwr as fl

    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(
            num_rounds=int(config.get("num_rounds", 50)),
            round_timeout=config.get("round_timeout_seconds"),
        ),
        strategy=strategy or create_fedavg(config),
    )
