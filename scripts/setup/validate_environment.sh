#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo_root"
uv run fedbrats check-environment
uv run fedbrats validate-config
uv run python -c 'import fl_brats, hydra, mlflow, monai, torch'
uv run pytest tests/test_config.py tests/test_environment.py
