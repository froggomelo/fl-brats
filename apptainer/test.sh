#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$script_dir/run.sh" info
"$script_dir/run.sh" validate-config
"$script_dir/run.sh" check-environment
"$script_dir/run.sh" python -m pytest tests
