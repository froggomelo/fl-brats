# fl-brats

Infrastructure-only skeleton for reproducible federated brain tumour segmentation research. It includes configuration, CLI, environment reporting, metadata tracking, and local/container/cluster launch templates. There is no model, dataset pipeline, training loop, evaluation, Flower client/server, aggregation, or federated runtime yet.

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
