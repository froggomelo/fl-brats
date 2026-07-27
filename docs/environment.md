# Environment

Use uv with the repository-local `.venv` and locked dependencies: `uv sync --group dev`. Cluster bootstrap uses only user-owned paths. The repository-local `.env` configures paths. By default, `FL_BRATS_DATA_ROOT=/datasets/$USER/brats`; this root contains `raw/brats2021`, `processed/brats2021_v1`, and fixed-quantity-skew partition manifests. Fixed Python, lockfiles, and reports support reproducibility.
