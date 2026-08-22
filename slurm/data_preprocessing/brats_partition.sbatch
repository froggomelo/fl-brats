#!/usr/bin/env bash
#SBATCH --job-name=brats-partition
#SBATCH --partition=${FL_BRATS_CPU_PARTITION:-cpu}
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/data_preprocessing_preprocessing/partition_%j.out
#SBATCH --error=logs/data_preprocessing_preprocessing/partition_%j.err
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
: "${FL_BRATS_DATA_ROOT:?set FL_BRATS_DATA_ROOT}"
root="$FL_BRATS_DATA_ROOT"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "image=$APPTAINER_IMAGE"
run_fedbrats() {
  apptainer exec --cleanenv \
    --bind "$root:$root" \
    "$APPTAINER_IMAGE" fedbrats "$@"
}
run_fedbrats create-split --index-path "$root/processed/brats2021_v1/dataset_index.csv" --manifest-path "$root/partitions/fixed_quantity_skew/split_manifest.json"
run_fedbrats create-partition --split-path "$root/partitions/fixed_quantity_skew/split_manifest.json" --output-dir "$root/partitions/fixed_quantity_skew"
run_fedbrats verify-manifests --partition-root "$root/partitions/fixed_quantity_skew"
