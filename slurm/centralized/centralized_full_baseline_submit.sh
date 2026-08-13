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

mkdir -p "$repo_root/logs/slurm"

partition="${FL_BRATS_PARTITION:-bigbatch}"
epochs="${CENTRALIZED_EPOCHS:-200}"
seeds="${FL_BRATS_SEEDS:-42 43 44}"
initial_weights_dir="${INITIAL_WEIGHTS_DIR:-artifacts/initial_weights}"

sbatch_args=(--partition="$partition")
if [[ -n "${FL_BRATS_NODELIST:-}" ]]; then
  sbatch_args+=(--nodelist="$FL_BRATS_NODELIST")
fi

dependency=()
for seed in $seeds; do
  weights_path="${INITIAL_WEIGHTS_PATH:-${initial_weights_dir}/unet_seed_${seed}.pt}"
  export_vars="ALL,FL_BRATS_PARTITION=${partition},CENTRALIZED_EPOCHS=${epochs},FL_BRATS_SEED=${seed},INITIAL_WEIGHTS_PATH=${weights_path}"
  if ((${#dependency[@]})); then
    job_id="$(sbatch --parsable "${dependency[@]}" "${sbatch_args[@]}" --export="$export_vars" "$repo_root/slurm/centralized/centralized_baseline.sbatch")"
  else
    job_id="$(sbatch --parsable "${sbatch_args[@]}" --export="$export_vars" "$repo_root/slurm/centralized/centralized_baseline.sbatch")"
  fi
  echo "centralized_seed_${seed}_job=$job_id epochs=$epochs weights=$weights_path"
  if [[ "${FL_BRATS_CHAIN_SEEDS:-0}" == "1" ]]; then
    dependency=(--dependency=afterok:"$job_id")
  fi
done
