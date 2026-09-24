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
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi
data_root="${FL_BRATS_DATA_ROOT:?set FL_BRATS_DATA_ROOT in .env}"
raw_root="${FL_BRATS_RAW_ROOT:-$data_root/raw/brats2021}"
[[ -d "$data_root" ]] || { echo "Configured dataset root does not exist: $data_root" >&2; exit 2; }
[[ -d "$raw_root" ]] || { echo "Configured raw dataset root does not exist: $raw_root" >&2; exit 2; }
export FL_BRATS_DATA_ROOT="$data_root"
export FL_BRATS_RAW_ROOT="$raw_root"
# Some nodes expose a GPU whose driver is broken; device=auto would then fall back
# to CPU silently. Exit 75 so the run can be resubmitted away from this node.
if [[ "${FL_BRATS_REQUIRE_GPU:-0}" == "1" ]] && ! nvidia-smi -L >/dev/null 2>&1; then
  echo "FL_BRATS_GPU_UNAVAILABLE host=$(hostname -s)" >&2
  exit 75
fi
if [[ "$use_apptainer" == "true" ]]; then
  [[ -f "$image" ]] || { echo "Configured Apptainer image does not exist: $image" >&2; exit 2; }
  apptainer_env=(
    --env "FL_BRATS_DATA_ROOT=$data_root"
    --env "FL_BRATS_RAW_ROOT=$raw_root"
    --env "FL_BRATS_OUTPUT_ROOT=${FL_BRATS_OUTPUT_ROOT:-$repo_root/outputs}"
    --env "FL_BRATS_LOG_ROOT=${FL_BRATS_LOG_ROOT:-$repo_root/logs}"
    --env "FL_BRATS_CACHE_ROOT=${FL_BRATS_CACHE_ROOT:-$repo_root/cache}"
    --env "FL_BRATS_CHECKPOINT_ROOT=${FL_BRATS_CHECKPOINT_ROOT:-$repo_root/checkpoints}"
  )
  # Images bake src/ in at build time. FL_BRATS_BIND_SOURCE=1 runs the checked-out
  # source instead (the image installs fl_brats editable from /workspace/src).
  source_bind=()
  if [[ "${FL_BRATS_BIND_SOURCE:-0}" == "1" ]]; then
    source_bind=(--bind "$repo_root/src:/workspace/src:ro")
  fi
  exec apptainer exec --nv "${apptainer_env[@]}" "${source_bind[@]}" \
    --bind "$repo_root:$repo_root" \
    --bind "$data_root:$data_root:ro" \
    --pwd "$repo_root" "$image" "$@"
fi
exec "$@"
