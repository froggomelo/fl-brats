#!/usr/bin/env bash
#SBATCH --job-name=fedbrats-sam-client
#SBATCH --partition=stampede
#SBATCH --output=logs/slurm/model_agnostic_uneven/client_%A_%a.out
#SBATCH --error=logs/slurm/model_agnostic_uneven/client_%A_%a.err
#SBATCH --nodes=1
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
: "${SAM_MODEL:?SAM_MODEL must be sam1 or sam2}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "image=$APPTAINER_IMAGE"
run_fedbrats() {
  apptainer exec --nv --cleanenv \
    --bind "$repo_root/src:/workspace/src:ro" \
    --bind "$repo_root/configs:/workspace/configs:ro" \
    "$APPTAINER_IMAGE" fedbrats "$@"
}
run_fedbrats verify-pretrained-model --model-config "configs/model/${SAM_MODEL}.yaml"
run_fedbrats "train-federated-${SAM_MODEL}"
