#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

mkdir -p "$repo_root/logs/slurm/federated/baseline"

partition="${FL_BRATS_PARTITION:-bigbatch}"
rounds="${FEDERATED_ROUNDS:-200}"
local_epochs="${FEDERATED_LOCAL_EPOCHS:-1}"
seeds="${FL_BRATS_SEEDS:-42 43 44}"
initial_weights_dir="${INITIAL_WEIGHTS_DIR:-artifacts/initial_weights}"

sbatch_args=(--partition="$partition")
if [[ -n "${FL_BRATS_NODELIST:-}" ]]; then
  sbatch_args+=(--nodelist="$FL_BRATS_NODELIST")
fi

dependency=()
for seed in $seeds; do
  run_id="${FEDERATED_RUN_PREFIX:-unet_baseline_full}_seed_${seed}_$(date +%Y%m%d_%H%M%S)"
  output_root="${FL_BRATS_OUTPUT_ROOT:-outputs}"
  run_dir="${FEDERATED_RUN_DIR:-${output_root}/federated/${run_id}}"
  weights_path="${INITIAL_WEIGHTS_PATH:-${initial_weights_dir}/unet_seed_${seed}.pt}"
  mkdir -p "$run_dir"

  export_vars="ALL,FL_BRATS_PARTITION=${partition},FEDERATED_RUN_ID=${run_id},FEDERATED_RUN_DIR=${run_dir},FEDERATED_ROUNDS=${rounds},FEDERATED_LOCAL_EPOCHS=${local_epochs},FL_BRATS_SEED=${seed},INITIAL_WEIGHTS_PATH=${weights_path}"

  if ((${#dependency[@]})); then
    server_job="$(sbatch --parsable "${dependency[@]}" "${sbatch_args[@]}" --export="$export_vars" "$repo_root/slurm/federated/baseline/federated_baseline_server.sbatch")"
  else
    server_job="$(sbatch --parsable "${sbatch_args[@]}" --export="$export_vars" "$repo_root/slurm/federated/baseline/federated_baseline_server.sbatch")"
  fi

  clients="${FEDERATED_CLIENTS:-4}"
  client_array_last=$((clients - 1))
  client_job="$(sbatch --parsable --dependency=after:"$server_job" --array="0-${client_array_last}" "${sbatch_args[@]}" --export="${export_vars},FEDERATED_SERVER_JOB_ID=${server_job},FEDERATED_CLIENTS=${clients}" "$repo_root/slurm/federated/baseline/federated_baseline_clients.sbatch")"
  cleanup_job="$(sbatch --parsable --dependency=afterany:"$client_job" "${sbatch_args[@]}" --export=ALL,FEDERATED_SERVER_JOB_ID="$server_job" --job-name=fedbrats-fedavg-cleanup --output=logs/slurm/federated/baseline/cleanup_%j.out --error=logs/slurm/federated/baseline/cleanup_%j.err --wrap='scancel "$FEDERATED_SERVER_JOB_ID" || true')"

  echo "federated_seed_${seed}_server_job=$server_job client_array_job=$client_job cleanup_job=$cleanup_job rounds=$rounds local_epochs=$local_epochs weights=$weights_path run_dir=$run_dir"
  if [[ "${FL_BRATS_CHAIN_SEEDS:-1}" == "1" ]]; then
    dependency=(--dependency=afterany:"$cleanup_job")
  fi
  sleep 1
done
