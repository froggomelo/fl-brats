#!/usr/bin/env bash

fl_brats_select_apptainer_image() {
  local repo_root="${1:?repo root required}"
  local partition="${2:-${SLURM_JOB_PARTITION:-}}"
  local key env_name selected

  if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
    export APPTAINER_IMAGE
    return 0
  fi

  partition="${partition%%,*}"
  partition="${partition:-default}"
  key="$(printf '%s' "$partition" | tr '[:lower:]-' '[:upper:]_')"
  env_name="FL_BRATS_IMAGE_${key}"
  selected="${!env_name:-}"

  if [[ -z "$selected" ]]; then
    case "$partition" in
      stampede)
        selected="${FL_BRATS_IMAGE_STAMPEDE:-$repo_root/apptainer/fl-brats-stampede.sif}"
        ;;
      batch)
        selected="${FL_BRATS_IMAGE_BATCH:-$repo_root/apptainer/fl-brats-batch.sif}"
        ;;
      *)
        selected="${FL_BRATS_IMAGE:-$repo_root/apptainer/fl-brats-batch.sif}"
        ;;
    esac
  fi

  export APPTAINER_IMAGE="$selected"
  export FL_BRATS_IMAGE="$selected"
}
