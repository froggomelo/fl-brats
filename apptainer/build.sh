#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$repo_root"
command -v apptainer >/dev/null || { echo "Apptainer is not available. Build on another Linux host or use a configured remote builder." >&2; exit 1; }
image="${APPTAINER_IMAGE:-$repo_root/apptainer/fl-brats.sif}"
if [[ -n "${APPTAINER_REMOTE:-}" ]]; then apptainer remote use "$APPTAINER_REMOTE"; fi
if apptainer build --help | grep -q -- '--remote'; then apptainer build --remote "$image" apptainer/fl-brats.def
else apptainer build "$image" apptainer/fl-brats.def || { echo "Unprivileged build was not permitted. Configure APPTAINER_REMOTE or build elsewhere; never use sudo." >&2; exit 1; }; fi
