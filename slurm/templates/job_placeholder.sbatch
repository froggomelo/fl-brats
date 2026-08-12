#!/usr/bin/env bash
# Generic infrastructure-only SLURM template; configure partition externally.
#SBATCH --job-name=fl-brats-infra
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
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
echo "partition=${SLURM_JOB_PARTITION:-unknown} image=$APPTAINER_IMAGE"
apptainer exec --cleanenv "$APPTAINER_IMAGE" fedbrats validate-config
