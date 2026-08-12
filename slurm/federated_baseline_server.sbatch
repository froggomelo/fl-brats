#!/usr/bin/env bash
# Server process for the distributed basic U-Net FedAvg run.
# Usually launched by: slurm/federated_baseline_submit.sh
#SBATCH --job-name=fedbrats-fedavg-srv
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/federated_server_%j.out
#SBATCH --error=logs/slurm/federated_server_%j.err

set -euo pipefail

trap 'echo "Received termination signal" >&2; exit 143' TERM INT

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
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
partition_root="${PARTITION_ROOT:-${data_root}/partitions/fixed_quantity_skew}"
run_id="${FEDERATED_RUN_ID:-unet_baseline_${SLURM_JOB_ID:-manual}}"
run_dir="${FEDERATED_RUN_DIR:-${output_root}/federated/${run_id}}"
experiment_name="${FEDERATED_EXPERIMENT_NAME:-federated_unet_baseline}"
model_config="${FEDERATED_MODEL:-unet_baseline}"
strategy_config="${FEDERATED_STRATEGY:-fedavg}"
read -r -a extra_overrides <<< "${FEDERATED_OVERRIDES:-}"
port="${FEDERATED_SERVER_PORT:-8080}"
host="$(hostname -f 2>/dev/null || hostname)"
address="${host}:${port}"

mkdir -p "$run_dir" "$log_root/slurm" "$cache_root" "$checkpoint_root"
for required in "$partition_root/client_0.json" "$partition_root/manifest.json" "$partition_root/split_manifest.json"; do
  [[ -f "$required" ]] || { echo "Required federated input missing: $required" >&2; exit 2; }
done
printf '%s\n' "$address" > "$run_dir/server_address.txt"

echo "hostname=$host job=${SLURM_JOB_ID:-unknown} address=$address"
echo "partition=${SLURM_JOB_PARTITION:-unknown} image=$APPTAINER_IMAGE"
echo "run_dir=$run_dir"

apptainer exec --cleanenv \
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
  fedbrats start-federated-server \
    --server-address "$address" \
    --run-dir "$run_dir" \
    "experiment_name=$experiment_name" \
    run_name="$run_id" \
    training=federated_client \
    "model=$model_config" \
    "federated=$strategy_config" \
    "data.partition_root=$partition_root" \
    "federated.num_rounds=${FEDERATED_ROUNDS:-50}" \
    "federated.num_clients=${FEDERATED_CLIENTS:-4}" \
    "federated.min_fit_clients=${FEDERATED_MIN_CLIENTS:-${FEDERATED_CLIENTS:-4}}" \
    "federated.min_evaluate_clients=${FEDERATED_MIN_CLIENTS:-${FEDERATED_CLIENTS:-4}}" \
    "federated.min_available_clients=${FEDERATED_MIN_CLIENTS:-${FEDERATED_CLIENTS:-4}}" \
    "federated.round_timeout_seconds=${FEDERATED_ROUND_TIMEOUT_SECONDS:-1800}" \
    "training.batch_size=${FEDERATED_BATCH_SIZE:-1}" \
    "training.pin_memory=false" \
    "${extra_overrides[@]}"
