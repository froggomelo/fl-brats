"""Flower server startup isolated from experiment orchestration."""

import json
from pathlib import Path
from typing import Any

from .strategy import create_fedavg


def start_server(
    address: str, config: dict[str, Any], strategy: Any = None, run_dir: Path | None = None
) -> None:
    import flwr as fl

    if strategy is None and run_dir is not None:
        from fl_brats.models.factory import create_model

        from .orchestration import _load_initial_weights
        from .strategy import create_research_fedavg

        run_dir.mkdir(parents=True, exist_ok=True)
        for relative in ("checkpoints", "metrics", "communication", "logs"):
            (run_dir / relative).mkdir(exist_ok=True)
        (run_dir / "resolved_config.yaml").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8"
        )
        model = create_model(config.get("model", {}))
        _load_initial_weights(model, config)
        strategy = create_research_fedavg(config, run_dir=run_dir, model=model)

    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(
            num_rounds=int(config.get("num_rounds", 50)),
            round_timeout=config.get("round_timeout_seconds"),
        ),
        strategy=strategy or create_fedavg(config),
    )
