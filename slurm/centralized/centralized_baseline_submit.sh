#!/usr/bin/env bash
set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

mkdir -p "$repo_root/logs/slurm/centralized"

partition="${FL_BRATS_PARTITION:-bigbatch}"
unset APPTAINER_IMAGE
source "$repo_root/apptainer/select_image.sh"
fl_brats_select_apptainer_image "$repo_root" "$partition"
[[ -f "$APPTAINER_IMAGE" ]] || {
  echo "Apptainer image missing for partition $partition: $APPTAINER_IMAGE" >&2
  exit 2
}

sbatch --partition="$partition" \
  --export=ALL,FL_BRATS_PARTITION="$partition",APPTAINER_IMAGE="$APPTAINER_IMAGE" \
  "$repo_root/slurm/centralized/centralized_baseline.sbatch"
