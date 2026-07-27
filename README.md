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

Hydra configuration lives in `configs/`; use overrides such as `uv run fedbrats show-config environment=debug`. Run `scripts/bootstrap.sh` for a no-global-install setup.

The default dataset root is `/datasets/$USER/brats`; see `data/README.md` for
the required BraTS 2021 raw, processed, and fixed-quantity-skew partition layout.

## Cluster

Apptainer is the primary cluster runtime and requires no sudo to execute. On a cluster, run `scripts/bootstrap_cluster.sh` with Python 3.11 available, then use user-owned bind paths.

```bash
apptainer exec fl-brats.sif fedbrats info
apptainer exec --nv --bind /path/to/data:/data:ro --bind /path/to/outputs:/outputs fl-brats.sif fedbrats check-environment
sbatch slurm/environment_check.sbatch
```

See `docs/` for configuration, environment, Apptainer, SLURM, cluster, architecture, and reproducibility guidance. Future modules will add data, models, training, federated orchestration, compression, and experiments while keeping this infrastructure boundary.
