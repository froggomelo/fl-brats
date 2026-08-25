#!/usr/bin/env bash

fl_brats_select_apptainer_image() {
  local repo_root="${1:?repo root required}"
  local partition="${2:-${SLURM_JOB_PARTITION:-}}"
  local selected

  partition="${partition%%,*}"
  partition="${partition:-default}"
  case "$partition" in
    stampede)
      selected="$repo_root/apptainer/fl-brats-stampede.sif"
      ;;
    batch|bigbatch)
      selected="$repo_root/apptainer/fl-brats-batch.sif"
      ;;
    *)
      selected="$repo_root/apptainer/fl-brats-batch.sif"
      ;;
  esac

  export APPTAINER_IMAGE="$selected"
}
