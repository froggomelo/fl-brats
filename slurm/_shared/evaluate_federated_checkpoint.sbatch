#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
[[ $# -eq 6 ]] || {
  echo "usage: $0 CONFIG CHECKPOINT OUTPUT_DIR CLIENT_PREFIX USE_APPTAINER IMAGE" >&2
  exit 2
}
config="$1"
checkpoint="$2"
output_dir="$3"
client_prefix="$4"
use_apptainer="$5"
image="$6"
repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
client_id="${client_prefix}${SLURM_ARRAY_TASK_ID:?array task ID is required}"
deadline=$((SECONDS + 600))
while [[ ! -s "$checkpoint" ]]; do
  (( SECONDS < deadline )) || { echo "timed out waiting for $checkpoint" >&2; exit 3; }
  sleep 2
done
exec "$repo_root/slurm/_shared/run_with_configured_environment.sh" \
  "$use_apptainer" "$image" \
  fedbrats evaluate-federated-checkpoint \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --client-id "$client_id" \
  --output-dir "$output_dir"
