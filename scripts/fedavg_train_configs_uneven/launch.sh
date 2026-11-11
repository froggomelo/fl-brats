#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG [launcher options]" >&2
  echo "configs are under configs/fedavg_train_configs_uneven/" >&2
  exit 2
fi
config="$1"
shift
exec python "$root/scripts/launch_experiment.py" "$config" "$@"
