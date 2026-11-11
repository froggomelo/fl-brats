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

partition="${FL_BRATS_DATA_PARTITION:-stampede}"
data_root="${FL_BRATS_DATA_ROOT:-/datasets/${USER}/brats}"
split_path="${FL_BRATS_SPLIT_PATH:-$data_root/partitions/fixed_quantity_skew/split_manifest.json}"
mkdir -p logs/data_preprocessing
[[ -f "$split_path" ]] || { echo "Dataset split missing: $split_path" >&2; exit 2; }

configs=(
  configs/fedavg_train_configs_uneven/client_sweeps/k3.yaml
  configs/fedavg_train_configs_uneven/client_sweeps/k5.yaml
  configs/fedavg_train_configs_uneven/client_sweeps/k8.yaml
  configs/fedavg_train_configs_uneven/local_epochs_sweeps/e5.yaml
  configs/fedavg_train_configs_uneven/modality_sweep/alpha_0_001.yaml
)
extra_args=(--split-path "$split_path")
[[ "${FL_BRATS_OVERWRITE_PARTITIONS:-0}" == "1" ]] && extra_args+=(--overwrite)

for config in "${configs[@]}"; do
  name="$(basename "${config%.yaml}")"
  job_id="$(sbatch --parsable --partition="$partition" \
    --job-name="flb-part-$name" \
    --output="logs/data_preprocessing/${name}_partition_%j.out" \
    --error="logs/data_preprocessing/${name}_partition_%j.err" \
    slurm/data_preprocessing/prepare_experiment_partition.sbatch \
    "$config" "${extra_args[@]}")"
  echo "config=$config job=$job_id partition=$partition"
done
