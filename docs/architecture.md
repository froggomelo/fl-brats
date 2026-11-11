# Architecture

This guide maps the `fl_brats` package and follows one federated run from configuration to results. Read it before changing code or adding a model, strategy, or compression method.

## Package Map

All code lives in `src/fl_brats/`. The `fedbrats` command in `cli.py` is the only entry point. Each subpackage below owns one concern and has no knowledge of the command line.

| Package | Responsibility |
|---|---|
| `cli.py` | Typer commands for setup, data preparation, training, evaluation, and reporting |
| `config/` | Hydra loading and Pydantic validation of the project configuration |
| `data/` | BraTS discovery, validation, indexing, preprocessing, splits, client partitions, modality masks, and data loaders |
| `models/` | Model factory and the U-Net, SegResNet, FedAMM, SAM 1, and SAM 2 models |
| `sam/` | 2D slicing, prompt, channel conversion, and volume reconstruction helpers for the SAM models |
| `losses/` | Multilabel Dice plus binary cross-entropy loss over the overlapping tumour regions |
| `training/` | One training and evaluation engine shared by centralised and federated runs, plus checkpoint I/O |
| `federated/` | Flower clients, strategies, orchestration, held-out evaluation, compression, and revalidation |
| `metrics/` | Dice, HD95, and sensitivity per tumour region, and run-level summaries |
| `comparison/` | Reports that compare a run against a reference baseline |
| `tracking/` | Local and MLflow metadata trackers |
| `utils/` | Paths, logging, git metadata, environment reports, and seeding |

## How a Federated Run Works

A run starts from one experiment YAML file. `scripts/launch_experiment.py` resolves it and submits one Slurm job per seed, or calls `fedbrats train-federated --config <file> --run-dir <dir>` directly. [Configuration](configuration.md) explains how the YAML is composed.

`federated/orchestration.py` drives the run. `run_local_federation` reads the client partition manifests under `partition_root` and builds one `ModelFedAvgClient` per client. It then creates the server model with `models.factory.create_model`, loads the per-seed initial weights, and restores state when the run resumes from a checkpoint. The resolved configuration is written to `resolved_config.yaml` inside the run directory.

`federated/strategy.py` builds the server strategy. FedAvg runs use `ResearchFedAvgStrategy` from `federated/research.py`. FedAMM runs use `FedAMMStrategy` from `federated/fedamm.py`, which aggregates each modality encoder separately. Both strategies write per-round metrics, save round checkpoints, track early stopping, and apply quantised broadcast when the configuration enables it.

Each client trains locally through `training/engine.py`. The engine is the same code the centralised baseline uses. Before upload the client compresses its update when quantisation (`federated/quantization.py`) or selective top-rho transmission (`federated/sparsification.py`) is enabled. `federated/communication.py` counts the payload bytes in both directions.

After each round `federated/evaluation.py` scores the global model on a fixed held-out validation cohort. That cohort never takes part in training. Checkpoint selection uses only this score. `federated/finalization.py` writes `summary.json` once the last round completes. Test-set reporting is a separate step described in [Evaluation](evaluation.md).

## Distributed Runs

`train-federated` runs the Flower server and all clients in one process, with each client on its own thread. The same code also runs across machines. `fedbrats start-federated-server` starts the server, and `fedbrats start-federated-client` starts one client per process. The Slurm jobs in `slurm/_shared/federated_server.sbatch` and `slurm/_shared/federated_client.sbatch` use these commands to place each client on its own GPU node. [Cluster](cluster.md) covers the job layout.

## Centralised Runs

`fedbrats train-centralized` trains one model on the pooled training split with the same engine, loss, and metrics. It provides the reference that the federated runs are compared against.

## Extending the Code

To add a model, write a module in `models/` that returns a `torch.nn.Module`, add a branch for its name in `models/factory.py`, and add a YAML file under `configs/model/`. [Models](models.md) describes the interface the engine expects.

To add a compression method, follow the pattern in `federated/sparsification.py`. The client encodes its update in `ModelFedAvgClient.fit`, the strategy decodes it before aggregation, and `federated/communication.py` must count the encoded bytes.

To add a metric, extend `metrics/segmentation.py`. Every model family reports through it, which keeps results comparable.

Add tests under `tests/` for any new behaviour and run `make test` and `make lint` before committing.
