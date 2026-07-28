#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$repo_root"
command -v apptainer >/dev/null || { echo "Apptainer is not available. Build on another Linux host or use a configured remote builder." >&2; exit 1; }
image="${APPTAINER_IMAGE:-$repo_root/apptainer/fl-brats.sif}"
if [[ -n "${APPTAINER_REMOTE:-}" ]]; then apptainer remote use "$APPTAINER_REMOTE"; fi
if apptainer build "$image" apptainer/fl-brats.def; then
  exit 0
fi
if [[ -n "${APPTAINER_REMOTE:-}" ]] && apptainer build --help | grep -q -- '--remote'; then
  apptainer build --remote "$image" apptainer/fl-brats.def
else
  echo "Local Apptainer build failed. Configure APPTAINER_REMOTE for a remote builder or build elsewhere; never use sudo." >&2
  exit 1
fi
