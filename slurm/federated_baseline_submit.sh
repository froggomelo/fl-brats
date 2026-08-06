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

server_job="$(
  sbatch --parsable --export=ALL,FEDERATED_RUN_ID="$run_id",FEDERATED_RUN_DIR="$run_dir" \
    slurm/federated_baseline_server.sbatch
)"
client_job="$(
  sbatch --parsable --export=ALL,FEDERATED_RUN_ID="$run_id",FEDERATED_RUN_DIR="$run_dir",FEDERATED_SERVER_JOB_ID="$server_job" \
    slurm/federated_baseline_clients.sbatch
)"

echo "server_job=$server_job"
echo "client_array_job=$client_job"
echo "run_dir=$run_dir"
