#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 CONFIG RUN_DIR PORT USE_APPTAINER IMAGE" >&2; exit 2; }
config="$1"
run_dir="$2"
port="$3"
use_apptainer="$4"
image="$5"
repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "$run_dir"
host="$(hostname -f 2>/dev/null || hostname)"
address="$host:$port"
printf '%s\n' "$address" > "$run_dir/server_address.txt"
exec "$repo_root/slurm/_shared/run_with_configured_environment.sh" "$use_apptainer" "$image" \
  fedbrats start-federated-server --config "$config" --run-dir "$run_dir" --server-address "$address"
