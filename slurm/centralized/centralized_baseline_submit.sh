#!/usr/bin/env bash
set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

export FL_BRATS_IMAGE_STAMPEDE=$repo_root/stampede.sif
export FL_BRATS_IMAGE_BATCH=$repo_root/batch.sif

sbatch --export=ALL slurm/centralized/centralized_baseline.sbatch
