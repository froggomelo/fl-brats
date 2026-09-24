#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
[[ $# -ge 2 ]] || {
  echo "usage: $0 RUN_LIST SPLIT [INDEX_PATH]" >&2
  exit 2
}
repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
mapfile -t runs < "$1"
run_dir="${runs[${SLURM_ARRAY_TASK_ID:-0}]:?run list has no entry for this task}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
args=(revalidate-federated-run --run-dir "$run_dir" --split "$2" --device cuda)
[[ -n "${3:-}" ]] && args+=(--index-path "$3")
exec "$repo_root/slurm/_shared/run_with_configured_environment.sh" \
  "${USE_APPTAINER:-true}" "${APPTAINER_IMAGE:-$repo_root/apptainer/fl-brats.sif}" \
  python -m fl_brats "${args[@]}"
