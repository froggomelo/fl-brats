#!/usr/bin/env bash
set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

output_root="${FL_BRATS_OUTPUT_ROOT:-outputs}"
run_id="${FEDERATED_RUN_ID:-unet_baseline_$(date +%Y%m%d_%H%M%S)}"
run_dir="${FEDERATED_RUN_DIR:-${output_root}/federated/${run_id}}"
mkdir -p "$run_dir"

# DataLoader multiprocessing is opt-in on this cluster because PyTorch tensor
# IPC can fail with "received 0 items of ancdata" under Apptainer.
: "${FEDERATED_WORKERS:=0}"

server_job="$(
  sbatch --parsable --export=ALL,FEDERATED_RUN_ID="$run_id",FEDERATED_RUN_DIR="$run_dir" \
    slurm/federated_baseline_server.sbatch
)"
client_job="$(
  sbatch --parsable --export=ALL,FEDERATED_RUN_ID="$run_id",FEDERATED_RUN_DIR="$run_dir",FEDERATED_SERVER_JOB_ID="$server_job" \
    slurm/federated_baseline_clients.sbatch
)"
cleanup_job="$(
  sbatch --parsable --dependency=afterany:"$client_job" \
    --export=ALL,FEDERATED_SERVER_JOB_ID="$server_job" \
    --job-name=fedbrats-fedavg-cleanup \
    --output=logs/slurm/federated_cleanup_%j.out \
    --error=logs/slurm/federated_cleanup_%j.err \
    --wrap='scancel "$FEDERATED_SERVER_JOB_ID" || true'
)"

echo "server_job=$server_job"
echo "client_array_job=$client_job"
echo "cleanup_job=$cleanup_job"
echo "run_dir=$run_dir"
