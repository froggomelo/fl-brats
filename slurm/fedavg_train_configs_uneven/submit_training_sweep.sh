#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

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

for config in "${configs[@]}"; do
  scripts/fedavg_train_configs_uneven/launch.sh "$config" \
    --mode slurm --partition "$partition" --image "$APPTAINER_IMAGE"
done
