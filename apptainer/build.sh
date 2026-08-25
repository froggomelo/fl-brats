#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$repo_root"
command -v apptainer >/dev/null || { echo "Apptainer is not available. Build on another Linux host or use a configured remote builder." >&2; exit 1; }
if [[ -n "${APPTAINER_REMOTE:-}" ]]; then apptainer remote use "$APPTAINER_REMOTE"; fi
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$repo_root/.apptainer-tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$repo_root/.apptainer-cache}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

build_one() {
  local image="$1"
  local definition="$2"
  rm -f "$image"
  if apptainer build --force "$image" "$definition"; then
    return 0
  fi
  if [[ -n "${APPTAINER_REMOTE:-}" ]] && apptainer build --help | grep -q -- '--remote'; then
    rm -f "$image"
    apptainer build --force --remote "$image" "$definition"
  else
    echo "Local Apptainer build failed for $image. Configure APPTAINER_REMOTE for a remote builder or build elsewhere; never use sudo." >&2
    return 1
  fi
}

build_one "$repo_root/apptainer/fl-brats-batch.sif" apptainer/fl-brats.def
build_one "$repo_root/apptainer/fl-brats-stampede.sif" apptainer/fl-brats-stampede.def
