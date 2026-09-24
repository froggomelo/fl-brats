# fl-brats

Operational research implementation for reproducible BraTS segmentation. It includes data preparation, a centralised U-Net baseline, deterministic partitions, missing-modality handling, Flower FedAvg clients/server, checkpointing, and payload accounting.

Run a local federation with a complete YAML experiment configuration:

```bash
fedbrats train-federated --config experiment.yaml --run-dir runs/fedavg_smoke
```

For physically distributed execution, start the server and then one process per client:

```bash
fedbrats start-federated-server --config experiment.yaml --server-address node01:8080
fedbrats start-federated-client --config experiment.yaml --client-id client_0 --server-address node01:8080
```

Run summaries include checkpoints, Dice metrics, and application-level serialized model payload estimates. Flower's public API does not expose lower-level network wire bytes, so transport headers are not claimed as measured.

## Local development

```bash
uv sync --group dev
uv run fedbrats info
uv run fedbrats validate-config
uv run fedbrats create-dirs
uv run pytest
uv run ruff check .
uv run mypy src
```

Hydra configuration lives in `configs/`; use overrides such as `uv run fedbrats show-config environment=debug`. Run `scripts/environment/bootstrap.sh` for a no-global-install setup.

The publication experiment matrix is defined by YAML in the mirrored folders
`configs/centralized/`, `configs/fedavg_uneven/`, `configs/fedamm_uneven/`, and
the other experiment-family directories. Matching launch scripts live under
`scripts/`. See [docs/experiments.md](docs/experiments.md) for launching,
partition preparation, output isolation, and metric artifacts.

The default dataset root is `/datasets/$USER/brats`; see `data/README.md` for
the required BraTS 2021 raw, processed, and fixed-quantity-skew partition layout.

## Cluster

Apptainer is the primary cluster runtime and requires no sudo to execute. On a cluster, run `scripts/environment/bootstrap_cluster.sh` with Python 3.11 available, then use user-owned bind paths.

```bash
bash apptainer/build.sh
FL_BRATS_PARTITION=stampede FL_BRATS_USE_GPU=1 ./apptainer/run.sh check-environment
sbatch slurm/environment/environment_check.sbatch
```

See `docs/` for configuration, environment, Apptainer, SLURM, cluster, architecture, and reproducibility guidance. Future modules will add data, models, training, federated orchestration, compression, and experiments while keeping this infrastructure boundary.

Federated model selection uses a fixed held-out validation cohort. See
[the validation and historical revalidation guide](docs/federated_validation.md)
for cohort semantics, diagnostics, resumable checkpoint search, separate final
test reporting, Slurm commands, and compute planning.
