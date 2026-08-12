#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/apptainer/select_image.sh"
fl_brats_select_apptainer_image "$repo_root" "${FL_BRATS_PARTITION:-${SLURM_JOB_PARTITION:-batch}}"
image="$APPTAINER_IMAGE"
: "${FL_BRATS_DATA_ROOT:?set FL_BRATS_DATA_ROOT}"
: "${FL_BRATS_OUTPUT_ROOT:?set FL_BRATS_OUTPUT_ROOT}"
: "${FL_BRATS_LOG_ROOT:?set FL_BRATS_LOG_ROOT}"
: "${FL_BRATS_CACHE_ROOT:?set FL_BRATS_CACHE_ROOT}"
: "${FL_BRATS_CHECKPOINT_ROOT:?set FL_BRATS_CHECKPOINT_ROOT}"
exec apptainer shell --cleanenv --bind "$FL_BRATS_DATA_ROOT:/data:ro" --bind "$FL_BRATS_OUTPUT_ROOT:/outputs" --bind "$FL_BRATS_LOG_ROOT:/logs" --bind "$FL_BRATS_CACHE_ROOT:/cache" --bind "$FL_BRATS_CHECKPOINT_ROOT:/checkpoints" "$image"
