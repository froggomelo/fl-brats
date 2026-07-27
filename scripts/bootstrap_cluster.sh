#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if ! command -v python3.11 >/dev/null; then echo "Python 3.11 is unavailable. Load your cluster Python module or ask administrators for a user-accessible Python 3.11." >&2; exit 1; fi
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
uv sync
uv run python -c 'import fl_brats, hydra, mlflow, monai, torch'
uv run fedbrats check-environment || true
echo "Cluster bootstrap completed without sudo. For an image, use apptainer/build.sh."
