#!/usr/bin/env bash
# Resources and per-run log paths are supplied from YAML by launch.py.
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$repo_root"
[[ $# -ge 3 ]] || { echo "Usage: $0 USE_APPTAINER IMAGE COMMAND..." >&2; exit 2; }
use_container="$1"
image="$2"
shift 2

echo "job_id=${SLURM_JOB_ID:-manual} host=$(hostname) command=$*"
if [[ "$use_container" == "true" ]]; then
  [[ -f "$image" ]] || { echo "Configured Apptainer image does not exist: $image" >&2; exit 2; }
  exec apptainer exec --nv --bind "$repo_root:$repo_root" --pwd "$repo_root" "$image" "$@"
fi
exec "$@"
