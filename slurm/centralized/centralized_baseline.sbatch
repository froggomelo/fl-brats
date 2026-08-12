#!/usr/bin/env bash
# Submit with: sbatch --export=ALL slurm/centralized/centralized_baseline.sbatch
# Overrides are supported, for example: sbatch --partition=batch --time=24:00:00 ...
#SBATCH --job-name=fedbrats-central
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/centralized_%j.out
#SBATCH --error=logs/slurm/centralized_%j.err

set -euo pipefail

trap 'echo "Received termination signal" >&2; exit 143' TERM INT

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
data_root="${FL_BRATS_DATA_ROOT:-/datasets/${USER}/brats}"
raw_root="${FL_BRATS_RAW_ROOT:-${data_root}/raw/brats2021}"
output_root="${FL_BRATS_OUTPUT_ROOT:-outputs}"
log_root="${FL_BRATS_LOG_ROOT:-logs}"
cache_root="${FL_BRATS_CACHE_ROOT:-cache}"
checkpoint_root="${FL_BRATS_CHECKPOINT_ROOT:-checkpoints}"
split_path="${SPLIT_MANIFEST_PATH:-${data_root}/partitions/fixed_quantity_skew/split_manifest.json}"
partition_path="${PARTITION_MANIFEST_PATH:-${data_root}/partitions/fixed_quantity_skew/manifest.json}"
index_path="${DATASET_INDEX_PATH:-${data_root}/processed/brats2021_v1/dataset_index.csv}"
weights_path="${INITIAL_WEIGHTS_PATH:-artifacts/initial_weights/unet_seed_42.pt}"

mkdir -p "$output_root" "$log_root/slurm" "$cache_root" "$checkpoint_root"
for required in "$index_path" "$split_path" "$partition_path" "$weights_path"; do
  [[ -f "$required" ]] || { echo "Required frozen input missing: $required" >&2; exit 2; }
done

echo "hostname=$(hostname) job=${SLURM_JOB_ID:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "image=$APPTAINER_IMAGE"
nvidia-smi || true

apptainer exec --nv --cleanenv \
  --env "FL_BRATS_RAW_ROOT=$raw_root" \
  --bind "$repo_root/src:/workspace/src:ro" \
  --bind "$repo_root/configs:/workspace/configs:ro" \
  --bind "$raw_root:$raw_root:ro" \
  --bind "$data_root:$data_root:ro" \
  --bind "$output_root:$output_root" \
  --bind "$log_root:$log_root" \
  --bind "$cache_root:$cache_root" \
  --bind "$cache_root:/cache" \
  --bind "$checkpoint_root:$checkpoint_root" \
  "$APPTAINER_IMAGE" \
  fedbrats train-centralized \
    --index-path "$index_path" \
    --split-path "$split_path" \
    --partition-path "$partition_path" \
    --initial-weights "$weights_path" \
    --output-root "$output_root/centralized" \
    --epochs "${CENTRALIZED_EPOCHS:-50}" \
    --batch-size "${CENTRALIZED_BATCH_SIZE:-1}" \
    --workers "${CENTRALIZED_WORKERS:-0}" \
    --no-pin-memory \
    --seed "${FL_BRATS_SEED:-42}" \
    --device cuda
