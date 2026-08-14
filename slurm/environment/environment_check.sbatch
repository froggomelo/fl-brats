#!/usr/bin/env bash
#SBATCH --job-name=fl-brats-env
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/environment/check_%j.out
#SBATCH --error=logs/slurm/environment/check_%j.err
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
echo "job=${SLURM_JOB_ID:-} node=${SLURMD_NODENAME:-} partition=${SLURM_JOB_PARTITION:-}"
echo "image=$APPTAINER_IMAGE"
apptainer exec --cleanenv "$APPTAINER_IMAGE" fedbrats info
apptainer exec --cleanenv "$APPTAINER_IMAGE" fedbrats check-environment
