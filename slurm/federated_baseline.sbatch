#!/usr/bin/env bash
# Submit with: sbatch --export=ALL slurm/federated_baseline.sbatch
# Optional overrides:
#   FEDERATED_ROUNDS=10 FEDERATED_CLIENTS=4 FEDERATED_BATCH_SIZE=1 sbatch --export=ALL slurm/federated_baseline.sbatch
#   FEDERATED_RUN_DIR=outputs/federated/my_run sbatch --export=ALL slurm/federated_baseline.sbatch
#SBATCH --job-name=fedbrats-fedavg
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/federated_%j.out
#SBATCH --error=logs/slurm/federated_%j.err

set -euo pipefail

trap 'echo "Received termination signal" >&2; exit 143' TERM INT

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

: "${APPTAINER_IMAGE:=${FL_BRATS_IMAGE:-}}"
: "${APPTAINER_IMAGE:?Set APPTAINER_IMAGE or FL_BRATS_IMAGE to the built fl-brats .sif image}"

data_root="${FL_BRATS_DATA_ROOT:-/datasets/${USER}/brats}"
raw_root="${FL_BRATS_RAW_ROOT:-${data_root}/raw/brats2021}"
output_root="${FL_BRATS_OUTPUT_ROOT:-outputs}"
log_root="${FL_BRATS_LOG_ROOT:-logs}"
cache_root="${FL_BRATS_CACHE_ROOT:-cache}"
checkpoint_root="${FL_BRATS_CHECKPOINT_ROOT:-checkpoints}"
partition_root="${PARTITION_ROOT:-${data_root}/partitions/fixed_quantity_skew}"
run_dir="${FEDERATED_RUN_DIR:-${output_root}/federated/unet_baseline_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$output_root/federated" "$log_root/slurm" "$cache_root" "$checkpoint_root"
for required in "$partition_root/client_0.json" "$partition_root/manifest.json" "$partition_root/split_manifest.json"; do
  [[ -f "$required" ]] || { echo "Required federated input missing: $required" >&2; exit 2; }
done

echo "hostname=$(hostname) job=${SLURM_JOB_ID:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "run_dir=$run_dir"
nvidia-smi || true

apptainer exec --nv --cleanenv \
  --env "FL_BRATS_RAW_ROOT=$raw_root" \
  --env "FL_BRATS_PIN_MEMORY=false" \
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
  fedbrats train-federated \
    --run-dir "$run_dir" \
    experiment=federated_unet_baseline \
    training=federated_client \
    model=unet_baseline \
    federated=fedavg \
    "data.partition_root=$partition_root" \
    "partition_root=$partition_root" \
    "num_rounds=${FEDERATED_ROUNDS:-50}" \
    "num_clients=${FEDERATED_CLIENTS:-4}" \
    "min_fit_clients=${FEDERATED_MIN_CLIENTS:-${FEDERATED_CLIENTS:-4}}" \
    "min_evaluate_clients=${FEDERATED_MIN_CLIENTS:-${FEDERATED_CLIENTS:-4}}" \
    "min_available_clients=${FEDERATED_MIN_CLIENTS:-${FEDERATED_CLIENTS:-4}}" \
    "local_batch_size=${FEDERATED_BATCH_SIZE:-1}" \
    "num_workers=${SLURM_CPUS_PER_TASK:-0}" \
    "pin_memory=false" \
    "device=cuda"
