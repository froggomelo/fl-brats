#!/usr/bin/env bash
#SBATCH --job-name=brats-preprocess
#SBATCH --partition=${FL_BRATS_CPU_PARTITION:-cpu}
#SBATCH --cpus-per-task=${FL_BRATS_CPUS:-4}
#SBATCH --mem=${FL_BRATS_MEM:-32G}
#SBATCH --output=logs/slurm/data/preprocess_%j.out
#SBATCH --error=logs/slurm/data/preprocess_%j.err
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
trap 'echo "termination received; completed subjects remain resumable"; exit 143' TERM INT
: "${FL_BRATS_DATA_ROOT:?set FL_BRATS_DATA_ROOT}"
raw_root="${FL_BRATS_RAW_ROOT:-$FL_BRATS_DATA_ROOT/raw/brats2021}"
container_root="/datasets/${USER}/brats"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "image=$APPTAINER_IMAGE"
apptainer exec --bind "$raw_root:$container_root/raw/brats2021:ro" --bind "$FL_BRATS_DATA_ROOT/processed:$container_root/processed:rw" "$APPTAINER_IMAGE" fedbrats preprocess-data --index-path "$container_root/processed/brats2021_v1/dataset_index.csv" --raw-root "$container_root/raw/brats2021" --output-dir "$container_root/processed/brats2021_v1" --resume
