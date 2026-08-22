#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

mkdir -p "$repo_root/logs/slurm/centralized"

# Set FL_BRATS_PARTITION to override the bigbatch default
# when invoking this script. The worker selects the matching Apptainer image:
# stampede -> fl-brats-stampede.sif; bigbatch/batch -> fl-brats-batch.sif.
partition="${FL_BRATS_PARTITION:-bigbatch}"
seeds="${FL_BRATS_SEEDS:-42 43 44}"
epochs="${CENTRALIZED_EPOCHS:-50}"
initial_weights_dir="${INITIAL_WEIGHTS_DIR:-artifacts/initial_weights}"

# Do not let an APPTAINER_IMAGE inherited from the login shell override the
# image required by the selected partition.
unset APPTAINER_IMAGE
source "$repo_root/apptainer/select_image.sh"
fl_brats_select_apptainer_image "$repo_root" "$partition"
[[ -f "$APPTAINER_IMAGE" ]] || {
  echo "Apptainer image missing for partition $partition: $APPTAINER_IMAGE" >&2
  exit 2
}

sbatch_args=(--partition="$partition" --exclude="${FL_BRATS_EXCLUDE_NODES:-mscluster25}")
if [[ -n "${FL_BRATS_NODELIST:-}" ]]; then
  sbatch_args+=(--nodelist="$FL_BRATS_NODELIST")
fi

dependency=()
for seed in $seeds; do
  weights_path="${INITIAL_WEIGHTS_PATH:-${initial_weights_dir}/unet_seed_${seed}.pt}"
  export_vars="ALL,FL_BRATS_PARTITION=${partition},APPTAINER_IMAGE=${APPTAINER_IMAGE},CENTRALIZED_EPOCHS=${epochs},FL_BRATS_SEED=${seed},INITIAL_WEIGHTS_PATH=${weights_path}"

  job_id="$(sbatch --parsable "${dependency[@]}" "${sbatch_args[@]}" \
    --export="$export_vars" \
    "$repo_root/slurm/centralized/centralized_baseline.sbatch")"

  echo "centralized_seed_${seed}_job=$job_id partition=$partition image=$APPTAINER_IMAGE epochs=$epochs weights=$weights_path"

  if [[ "${FL_BRATS_CHAIN_SEEDS:-0}" == "1" ]]; then
    dependency=(--dependency=afterok:"$job_id")
  else
    dependency=()
  fi
done
