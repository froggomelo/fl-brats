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
# The address is published before the server's Python process is listening; it
# first loads weights and prepares held-out validation. Flower clients give up
# after a short run of refused connections, so wait for the port to accept.
host="${address%:*}"
port="${address##*:}"
deadline=$((SECONDS + 1800))
until (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; do
  (( SECONDS < deadline )) || { echo "timed out waiting for server at $address" >&2; exit 3; }
  sleep 5
done
client_id="client_${SLURM_ARRAY_TASK_ID:-0}"
exec "$repo_root/slurm/_shared/run_with_configured_environment.sh" "$use_apptainer" "$image" \
  fedbrats start-federated-client --config "$config" --client-id "$client_id" --server-address "$address"
