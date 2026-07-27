#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${APPTAINER_IMAGE:-$repo_root/apptainer/fl-brats.sif}"
command -v apptainer >/dev/null || { echo "Apptainer unavailable." >&2; exit 1; }
[[ -f "$image" ]] || { echo "Image missing: $image" >&2; exit 1; }
for variable in FL_BRATS_DATA_ROOT FL_BRATS_OUTPUT_ROOT FL_BRATS_LOG_ROOT FL_BRATS_CACHE_ROOT FL_BRATS_CHECKPOINT_ROOT; do
  value="${!variable:-}"; [[ -n "$value" ]] || { echo "$variable must be set" >&2; exit 1; }
done
mkdir -p "$FL_BRATS_OUTPUT_ROOT" "$FL_BRATS_LOG_ROOT" "$FL_BRATS_CACHE_ROOT" "$FL_BRATS_CHECKPOINT_ROOT"
[[ -d "$FL_BRATS_DATA_ROOT" ]] || { echo "Data root missing: $FL_BRATS_DATA_ROOT" >&2; exit 1; }
args=(exec --cleanenv --bind "$FL_BRATS_DATA_ROOT:/data:ro" --bind "$FL_BRATS_OUTPUT_ROOT:/outputs" --bind "$FL_BRATS_LOG_ROOT:/logs" --bind "$FL_BRATS_CACHE_ROOT:/cache" --bind "$FL_BRATS_CHECKPOINT_ROOT:/checkpoints")
[[ "${FL_BRATS_USE_GPU:-0}" == "1" ]] && args+=(--nv)
exec apptainer "${args[@]}" "$image" fedbrats "$@"
