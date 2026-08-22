#!/usr/bin/env bash
# Create deterministic initial U-Net weights used by controlled baselines.
# Submit with: sbatch --export=ALL slurm/fedavg_uneven/create_initial_weights.sbatch
# Optional overrides:
#   INITIAL_WEIGHTS_DIR=artifacts/initial_weights
#   INITIAL_WEIGHTS_SEED=42
#   MODEL_CONFIG=configs/model/unet_baseline.yaml
#SBATCH --job-name=fedbrats-initw
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/fedavg_uneven/init_weights_%j.out
#SBATCH --error=logs/slurm/fedavg_uneven/init_weights_%j.err

set -euo pipefail

trap 'echo "Initial weight creation interrupted" >&2; exit 143' TERM INT

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$repo_root"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

# Use the partition-specific image even if the submitting shell retained a
# legacy APPTAINER_IMAGE value.
unset APPTAINER_IMAGE
source "$repo_root/apptainer/select_image.sh"
fl_brats_select_apptainer_image "$repo_root"
[[ -f "$APPTAINER_IMAGE" ]] || { echo "Apptainer image missing: $APPTAINER_IMAGE" >&2; exit 2; }

model_config="${MODEL_CONFIG:-configs/model/unet_baseline.yaml}"
output_dir="${INITIAL_WEIGHTS_DIR:-artifacts/initial_weights}"
seed="${INITIAL_WEIGHTS_SEED:-42}"

mkdir -p logs/fedavg_uneven "$output_dir"

[[ -f "$model_config" ]] || { echo "Model config missing: $model_config" >&2; exit 2; }

echo "hostname=$(hostname) job=${SLURM_JOB_ID:-unknown}"
echo "repository=$repo_root"
echo "image=$APPTAINER_IMAGE"
echo "model_config=$model_config"
echo "output_dir=$output_dir"
echo "seed=$seed"

apptainer exec --cleanenv \
  --bind "$repo_root:$repo_root" \
  "$APPTAINER_IMAGE" \
  fedbrats create-initial-weights \
    --model-config "$model_config" \
    --output-dir "$output_dir" \
    --seed "$seed"

weights_path="$output_dir/unet_seed_${seed}.pt"
metadata_path="$output_dir/unet_seed_${seed}.json"
[[ -f "$weights_path" ]] || { echo "Expected weights were not created: $weights_path" >&2; exit 3; }
[[ -f "$metadata_path" ]] || { echo "Expected metadata was not created: $metadata_path" >&2; exit 3; }

ls -lh "$weights_path" "$metadata_path"
echo "completed=$(date --iso-8601=seconds)"
