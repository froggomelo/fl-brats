#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
[[ -d .venv ]] || { echo "Missing .venv; run scripts/setup/bootstrap.sh first." >&2; exit 1; }
exec uv run fedbrats "$@"
