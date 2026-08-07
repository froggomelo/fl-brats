#!/usr/bin/env bash
set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

sbatch --export=ALL slurm/centralized_baseline.sbatch
