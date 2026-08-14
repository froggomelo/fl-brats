#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

output_root="${FL_BRATS_OUTPUT_ROOT:-outputs}"
training_partition="${FL_BRATS_SLURM_PARTITION:-stampede}"
run_id="${FEDERATED_RUN_ID:-unet_baseline_$(date +%Y%m%d_%H%M%S)}"
run_dir="${FEDERATED_RUN_DIR:-${output_root}/federated/${run_id}}"
mkdir -p "$run_dir" "$repo_root/logs/slurm/federated/baseline"

# DataLoader multiprocessing is opt-in on this cluster because PyTorch tensor
# IPC can fail with "received 0 items of ancdata" under Apptainer.
: "${FEDERATED_WORKERS:=0}"
: "${FEDERATED_CLIENTS:=4}"
if ! [[ "$FEDERATED_CLIENTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "FEDERATED_CLIENTS must be a positive integer, got: $FEDERATED_CLIENTS" >&2
  exit 2
fi
client_array_last=$((FEDERATED_CLIENTS - 1))

server_job="$(
  sbatch --parsable --partition="$training_partition" \
    --export=ALL,FL_BRATS_REPO_ROOT="$repo_root",FEDERATED_RUN_ID="$run_id",FEDERATED_RUN_DIR="$run_dir" \
    "$repo_root/slurm/federated/baseline/federated_baseline_server.sbatch"
)"
client_job="$(
  sbatch --parsable --partition="$training_partition" --array="0-${client_array_last}" \
    --export=ALL,FL_BRATS_REPO_ROOT="$repo_root",FEDERATED_RUN_ID="$run_id",FEDERATED_RUN_DIR="$run_dir",FEDERATED_SERVER_JOB_ID="$server_job",FEDERATED_CLIENTS="$FEDERATED_CLIENTS" \
    "$repo_root/slurm/federated/baseline/federated_baseline_clients.sbatch"
)"
cleanup_job="$(
  sbatch --parsable --partition="$training_partition" --dependency=afterany:"$client_job" \
    --export=ALL,FEDERATED_SERVER_JOB_ID="$server_job" \
    --job-name=fedbrats-fedavg-cleanup \
    --output=logs/slurm/federated/baseline/cleanup_%j.out \
    --error=logs/slurm/federated/baseline/cleanup_%j.err \
    --wrap='scancel "$FEDERATED_SERVER_JOB_ID" || true'
)"

echo "server_job=$server_job"
echo "client_array_job=$client_job"
echo "cleanup_job=$cleanup_job"
echo "run_dir=$run_dir"
