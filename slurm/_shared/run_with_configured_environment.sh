#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 3 ]] || {
  echo "usage: $0 USE_APPTAINER IMAGE COMMAND..." >&2
  exit 2
}
use_apptainer="$1"
image="$2"
shift 2
repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$repo_root"
if [[ "$use_apptainer" == "true" ]]; then
  [[ -f "$image" ]] || { echo "Configured Apptainer image does not exist: $image" >&2; exit 2; }
  exec apptainer exec --nv --bind "$repo_root:$repo_root" --pwd "$repo_root" "$image" "$@"
fi
exec "$@"
