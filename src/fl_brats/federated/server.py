"""Flower server startup isolated from experiment orchestration."""

import json
import time
from pathlib import Path
from typing import Any

from .evaluation import HeldOutEvaluator
from .finalization import finalize_federation
from .orchestration import load_resume_state, rounds_to_run
from .strategy import create_fedavg


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def start_server(
    address: str, config: dict[str, Any], strategy: Any = None, run_dir: Path | None = None
) -> None:
    import flwr as fl

    if strategy is None and run_dir is None:
        raise ValueError("a new run_dir is required for held-out evaluation and checkpoints")
    if strategy is None and run_dir is not None:
        from fl_brats.models.factory import create_model

        from .orchestration import _load_initial_weights
        from .strategy import create_research_fedavg

        # The Slurm bootstrap publishes the address before starting Python.
        if run_dir.exists() and any(p.name != "server_address.txt" for p in run_dir.iterdir()):
            raise ValueError("server run directory already contains experiment artifacts")
        run_dir.mkdir(parents=True, exist_ok=True)
        for relative in ("checkpoints", "metrics", "communication", "logs"):
            (run_dir / relative).mkdir(exist_ok=True)
        (run_dir / "resolved_config.yaml").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8"
        )
        model = create_model(config.get("model", {}))
        config = dict(config)
        config["initial_weight_hash"] = _load_initial_weights(model, config)
        resume_state = load_resume_state(model, config, run_dir)
        evaluator = HeldOutEvaluator(config)
        evaluator.prepare()
        strategy = create_research_fedavg(
            config,
            run_dir=run_dir,
            model=model,
            evaluate_fn=evaluator.callback(model, run_dir),
            initial_state=resume_state,
            round_offset=resume_state.completed_round if resume_state else 0,
        )
    else:
        resume_state = None

    started = time.monotonic()
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(
            num_rounds=rounds_to_run(config, resume_state),
            round_timeout=config.get("round_timeout_seconds"),
        ),
        strategy=strategy or create_fedavg(config),
    )
    if run_dir is not None and strategy is not None and hasattr(strategy, "state"):
        evaluator = HeldOutEvaluator(config)
        finalize_federation(strategy, evaluator, config, run_dir, time.monotonic() - started)
