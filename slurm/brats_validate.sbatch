#!/usr/bin/env bash
#SBATCH --job-name=brats-validate
#SBATCH --partition=${FL_BRATS_CPU_PARTITION:-cpu}
#SBATCH --cpus-per-task=${FL_BRATS_CPUS:-4}
#SBATCH --mem=${FL_BRATS_MEM:-16G}
#SBATCH --output=logs/brats_validate_%j.out
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi
echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"
: "${FL_BRATS_DATA_ROOT:?set FL_BRATS_DATA_ROOT}"
: "${FL_BRATS_IMAGE:?set FL_BRATS_IMAGE}"
container_root="/datasets/${USER}/brats"
apptainer exec --bind "$FL_BRATS_DATA_ROOT:$container_root:ro" "$FL_BRATS_IMAGE" fedbrats validate-data --root "$container_root/raw/brats2021" --strict
