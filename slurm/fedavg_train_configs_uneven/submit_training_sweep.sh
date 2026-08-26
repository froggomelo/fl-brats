#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

initial_afterany=""
if [[ ${1:-} == "--afterany" ]]; then
  [[ -n ${2:-} ]] || {
    echo "usage: $0 [--afterany JOBID]" >&2
    exit 2
  }
  initial_afterany="$2"
  shift 2
fi
[[ $# -eq 0 ]] || {
  echo "usage: $0 [--afterany JOBID]" >&2
  exit 2
}

if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

partition="${FL_BRATS_TRAINING_PARTITION:-bigbatch}"
unset APPTAINER_IMAGE
# shellcheck disable=SC1091
source "$repo_root/apptainer/select_image.sh"
fl_brats_select_apptainer_image "$repo_root" "$partition"
[[ -f "$APPTAINER_IMAGE" ]] || {
  echo "Apptainer image missing for $partition: $APPTAINER_IMAGE" >&2
  exit 2
}

configs=(
  configs/fedavg_train_configs_uneven/client_sweeps/k3.yaml
  configs/fedavg_train_configs_uneven/client_sweeps/k5.yaml
  configs/fedavg_train_configs_uneven/client_sweeps/k8.yaml
  configs/fedavg_train_configs_uneven/local_epochs_sweeps/e5.yaml
  configs/fedavg_train_configs_uneven/local_epochs_sweeps/e10.yaml
  configs/fedavg_train_configs_uneven/mini_batch_sweeps/b4.yaml
  configs/fedavg_train_configs_uneven/mini_batch_sweeps/b8.yaml
)

previous_cleanup_job="$initial_afterany"
for config in "${configs[@]}"; do
  launcher_args=(
    --mode slurm
    --partition "$partition"
    --image "$APPTAINER_IMAGE"
  )
  if [[ -n "$previous_cleanup_job" ]]; then
    launcher_args+=(--afterany "$previous_cleanup_job")
  fi

  submission="$(
    python3 scripts/launch_experiment.py "$config" "${launcher_args[@]}"
  )"
  printf '%s\n' "$submission"
  previous_cleanup_job="$(
    python3 -c 'import json, sys; print(json.load(sys.stdin)[-1]["jobs"]["cleanup"])' \
      <<<"$submission"
  )"
done
