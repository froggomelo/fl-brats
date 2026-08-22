#!/usr/bin/env bash
#SBATCH --job-name=brats-validate
#SBATCH --partition=${FL_BRATS_CPU_PARTITION:-cpu}
#SBATCH --cpus-per-task=${FL_BRATS_CPUS:-4}
#SBATCH --mem=${FL_BRATS_MEM:-16G}
#SBATCH --output=logs/data_preprocessing_preprocessing/validate_%j.out
#SBATCH --error=logs/data_preprocessing_preprocessing/validate_%j.err
set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi
source "$repo_root/apptainer/select_image.sh"
fl_brats_select_apptainer_image "$repo_root"
[[ -f "$APPTAINER_IMAGE" ]] || { echo "Apptainer image missing: $APPTAINER_IMAGE" >&2; exit 2; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "image=$APPTAINER_IMAGE"
: "${FL_BRATS_DATA_ROOT:?set FL_BRATS_DATA_ROOT}"
raw_root="${FL_BRATS_RAW_ROOT:-$FL_BRATS_DATA_ROOT/raw/brats2021}"
container_root="/datasets/${USER}/brats"
apptainer exec --bind "$raw_root:$container_root/raw/brats2021:ro" "$APPTAINER_IMAGE" fedbrats validate-data --root "$container_root/raw/brats2021" --strict
