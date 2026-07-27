#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ -n "${FL_BRATS_PYTHON_MODULE:-}" ]]; then
  command -v module >/dev/null || { echo "Environment modules are unavailable; cannot load FL_BRATS_PYTHON_MODULE." >&2; exit 1; }
  module load "$FL_BRATS_PYTHON_MODULE"
fi
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
# Do not depend on the cluster's system Python: uv installs a managed CPython
# into the user's cache when python3.11 is absent. No sudo or module is needed.
if ! command -v python3.11 >/dev/null; then
  echo "System Python 3.11 unavailable; installing uv-managed CPython 3.11."
  uv python install 3.11
fi
uv sync --group dev --python 3.11
uv run --python 3.11 python -c 'import fl_brats, hydra, mlflow, monai, torch'
uv run --python 3.11 fedbrats check-environment || true
echo "Cluster bootstrap completed with Python 3.11 managed by uv when required."
