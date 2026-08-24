#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 CONFIG RUN_DIR USE_APPTAINER IMAGE" >&2; exit 2; }
config="$1"
run_dir="$2"
use_apptainer="$3"
image="$4"
repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
address_file="$run_dir/server_address.txt"
deadline=$((SECONDS + 600))
while [[ ! -s "$address_file" ]]; do
  (( SECONDS < deadline )) || { echo "timed out waiting for $address_file" >&2; exit 3; }
  sleep 2
done
address="$(<"$address_file")"
client_id="client_${SLURM_ARRAY_TASK_ID:-0}"
exec "$repo_root/slurm/_shared/run_with_configured_environment.sh" "$use_apptainer" "$image" \
  fedbrats start-federated-client --config "$config" --client-id "$client_id" --server-address "$address"
