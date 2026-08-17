#!/usr/bin/env bash
set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

mkdir -p "$repo_root/logs/slurm/centralized" "$repo_root/logs/slurm/federated/baseline"

partition="${FL_BRATS_PARTITION:-stampede}"
seeds="${FL_BRATS_SEEDS:-42 43 44}"
initial_weights_dir="${INITIAL_WEIGHTS_DIR:-artifacts/initial_weights}"
centralized_epochs="${CENTRALIZED_EPOCHS:-200}"
federated_rounds="${FEDERATED_ROUNDS:-200}"
federated_local_epochs="${FEDERATED_LOCAL_EPOCHS:-1}"
federated_clients="${FEDERATED_CLIENTS:-4}"
output_root="${FL_BRATS_OUTPUT_ROOT:-outputs}"

unset APPTAINER_IMAGE
source "$repo_root/apptainer/select_image.sh"
fl_brats_select_apptainer_image "$repo_root" "$partition"
[[ -f "$APPTAINER_IMAGE" ]] || {
  echo "Apptainer image missing for partition $partition: $APPTAINER_IMAGE" >&2
  exit 2
}

if ! [[ "$federated_clients" =~ ^[1-9][0-9]*$ ]]; then
  echo "FEDERATED_CLIENTS must be a positive integer, got: $federated_clients" >&2
  exit 2
fi

sbatch_args=(--partition="$partition")
if [[ -n "${FL_BRATS_NODELIST:-}" ]]; then
  sbatch_args+=(--nodelist="$FL_BRATS_NODELIST")
fi
gpu_sbatch_args=("${sbatch_args[@]}" --exclude="${FL_BRATS_EXCLUDE_NODES:-mscluster25}")

submit_initial_weights() {
  local seed="$1"
  local weights_path="${initial_weights_dir}/unet_seed_${seed}.pt"
  shift
  local dependency=("$@")

  if [[ -f "$weights_path" && "${FL_BRATS_RECREATE_INITIAL_WEIGHTS:-0}" != "1" ]]; then
    echo "initial_weights_seed_${seed}=existing path=$weights_path"
    return 0
  fi

  sbatch --parsable "${dependency[@]}" "${sbatch_args[@]}" \
    --export="ALL,FL_BRATS_PARTITION=${partition},APPTAINER_IMAGE=${APPTAINER_IMAGE},INITIAL_WEIGHTS_DIR=${initial_weights_dir},INITIAL_WEIGHTS_SEED=${seed}" \
    "$repo_root/slurm/federated/baseline/create_initial_weights.sbatch"
}

submit_centralized() {
  local seed="$1"
  local weights_path="${initial_weights_dir}/unet_seed_${seed}.pt"
  shift
  local dependency=("$@")

  sbatch --parsable "${dependency[@]}" "${gpu_sbatch_args[@]}" \
    --export="ALL,FL_BRATS_PARTITION=${partition},APPTAINER_IMAGE=${APPTAINER_IMAGE},CENTRALIZED_EPOCHS=${centralized_epochs},FL_BRATS_SEED=${seed},INITIAL_WEIGHTS_PATH=${weights_path}" \
    "$repo_root/slurm/centralized/centralized_baseline.sbatch"
}

submit_federated() {
  local seed="$1"
  local weights_path="${initial_weights_dir}/unet_seed_${seed}.pt"
  shift
  local dependency=("$@")
  local run_id="${FEDERATED_RUN_PREFIX:-unet_baseline_full}_seed_${seed}_$(date +%Y%m%d_%H%M%S)"
  local run_dir="${FEDERATED_RUN_DIR:-${output_root}/federated/${run_id}}"
  local client_array_last=$((federated_clients - 1))
  local export_vars
  local server_job
  local client_job
  local cleanup_job

  mkdir -p "$run_dir"
  export_vars="ALL,FL_BRATS_PARTITION=${partition},APPTAINER_IMAGE=${APPTAINER_IMAGE},FEDERATED_RUN_ID=${run_id},FEDERATED_RUN_DIR=${run_dir},FEDERATED_ROUNDS=${federated_rounds},FEDERATED_LOCAL_EPOCHS=${federated_local_epochs},FEDERATED_CLIENTS=${federated_clients},FL_BRATS_SEED=${seed},INITIAL_WEIGHTS_PATH=${weights_path}"

  server_job="$(sbatch --parsable "${dependency[@]}" "${sbatch_args[@]}" \
    --export="$export_vars" \
    "$repo_root/slurm/federated/baseline/federated_baseline_server.sbatch")"

  client_job="$(sbatch --parsable --dependency=after:"$server_job" --array="0-${client_array_last}" "${gpu_sbatch_args[@]}" \
    --export="${export_vars},FEDERATED_SERVER_JOB_ID=${server_job}" \
    "$repo_root/slurm/federated/baseline/federated_baseline_clients.sbatch")"

  cleanup_job="$(sbatch --parsable --dependency=afterany:"$client_job" "${sbatch_args[@]}" \
    --export="ALL,FEDERATED_SERVER_JOB_ID=${server_job}" \
    --job-name=fedbrats-fedavg-cleanup \
    --output=logs/slurm/federated/baseline/cleanup_%j.out \
    --error=logs/slurm/federated/baseline/cleanup_%j.err \
    --wrap='scancel "$FEDERATED_SERVER_JOB_ID" || true')"

  submitted_federated_cleanup_job="$cleanup_job"
  echo "federated_seed_${seed}_server_job=$server_job client_array_job=$client_job cleanup_job=$cleanup_job partition=$partition image=$APPTAINER_IMAGE rounds=$federated_rounds local_epochs=$federated_local_epochs run_dir=$run_dir"
}

seed_dependency=()
submitted_federated_cleanup_job=""
for seed in $seeds; do
  weights_job="$(submit_initial_weights "$seed" "${seed_dependency[@]}")"
  if [[ "$weights_job" == initial_weights_seed_* ]]; then
    weight_dependency=("${seed_dependency[@]}")
  else
    echo "initial_weights_seed_${seed}_job=$weights_job"
    weight_dependency=(--dependency=afterok:"$weights_job")
  fi

  centralized_job="$(submit_centralized "$seed" "${weight_dependency[@]}")"
  echo "centralized_seed_${seed}_job=$centralized_job epochs=$centralized_epochs"

  fed_dependency=("${weight_dependency[@]}")
  if [[ "${FL_BRATS_CHAIN_CENTRALIZED_BEFORE_FEDERATED:-0}" == "1" ]]; then
    fed_dependency=(--dependency=afterok:"$centralized_job")
  fi
  submit_federated "$seed" "${fed_dependency[@]}"

  if [[ "${FL_BRATS_CHAIN_SEEDS:-1}" == "1" ]]; then
    seed_dependency=(--dependency=afterok:"$centralized_job",afterany:"$submitted_federated_cleanup_job")
  else
    seed_dependency=()
  fi
done
