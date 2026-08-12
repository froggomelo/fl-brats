#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$repo_root"
command -v apptainer >/dev/null || { echo "Apptainer is not available. Build on another Linux host or use a configured remote builder." >&2; exit 1; }
if [[ -n "${APPTAINER_REMOTE:-}" ]]; then apptainer remote use "$APPTAINER_REMOTE"; fi

build_one() {
  local image="$1"
  local definition="$2"
  if apptainer build "$image" "$definition"; then
    return 0
  fi
  if [[ -n "${APPTAINER_REMOTE:-}" ]] && apptainer build --help | grep -q -- '--remote'; then
    apptainer build --remote "$image" "$definition"
  else
    echo "Local Apptainer build failed for $image. Configure APPTAINER_REMOTE for a remote builder or build elsewhere; never use sudo." >&2
    return 1
  fi
}

build_one "${FL_BRATS_IMAGE_BATCH:-$repo_root/apptainer/fl-brats-batch.sif}" apptainer/fl-brats.def
build_one "${FL_BRATS_IMAGE_STAMPEDE:-$repo_root/apptainer/fl-brats-stampede.sif}" apptainer/fl-brats-stampede.def
