#!/usr/bin/env bash
# Four-client array for the distributed basic U-Net FedAvg run.
# Usually launched by: slurm/federated_baseline_submit.sh
#SBATCH --job-name=fedbrats-fedavg-cli
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/federated_client_%A_%a.out
#SBATCH --error=logs/slurm/federated_client_%A_%a.err

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
run_id="${FEDERATED_RUN_ID:-unet_baseline_manual}"
run_dir="${FEDERATED_RUN_DIR:-${output_root}/federated/${run_id}}"
client_index="${SLURM_ARRAY_TASK_ID:-0}"
client_id="client_${client_index}"
address_file="$run_dir/server_address.txt"

mkdir -p "$log_root/slurm" "$cache_root" "$checkpoint_root"
required_client="$partition_root/${client_id}.json"
[[ -f "$required_client" ]] || { echo "Required client manifest missing: $required_client" >&2; exit 2; }

deadline=$((SECONDS + ${FEDERATED_SERVER_WAIT_SECONDS:-300}))
while [[ ! -s "$address_file" ]]; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for server address file: $address_file" >&2
    exit 3
  fi
  sleep 2
done
server_address="$(<"$address_file")"

echo "hostname=$(hostname) job=${SLURM_JOB_ID:-unknown} client_id=$client_id server=$server_address"
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
  fedbrats start-federated-client \
    --client-id "$client_id" \
    --server-address "$server_address" \
    experiment_name=federated_unet_baseline \
    run_name="$run_id" \
    training=federated_client \
    model=unet_baseline \
    federated=fedavg \
    "data.partition_root=$partition_root" \
    "federated.num_rounds=${FEDERATED_ROUNDS:-50}" \
    "federated.num_clients=${FEDERATED_CLIENTS:-4}" \
    "training.batch_size=${FEDERATED_BATCH_SIZE:-1}" \
    "training.num_workers=${FEDERATED_WORKERS:-0}" \
    "training.pin_memory=false" \
    "device=cuda"
