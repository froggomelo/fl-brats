#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

[[ -f data/partitions/fedavg_uneven/manifest.json ]] || {
  echo "Uneven partition is missing: data/partitions/fedavg_uneven/manifest.json" >&2
  echo "Run slurm/data_preprocessing/prepare_fedavg_baseline_partitions_submit.sh first." >&2
  exit 2
}

exec python3 scripts/launch_experiment.py configs/fedavg_uneven/baseline.yaml "$@"
