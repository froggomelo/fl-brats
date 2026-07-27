#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if ! python3.11 --version >/dev/null 2>&1; then echo "Python 3.11 is required; install or load it first." >&2; exit 1; fi
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
uv sync --group dev
uv run pre-commit install
if [[ ! -e .env ]]; then cp .env.example .env; fi
echo "Ready. Try: uv run fedbrats info && uv run pytest"
