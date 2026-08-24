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
split_path="${FL_BRATS_SPLIT_PATH:-${data_root}/partitions/fixed_quantity_skew/split_manifest.json}"
mkdir -p logs/data_preprocessing

[[ -f "$split_path" ]] || {
  echo "Dataset split missing: $split_path" >&2
  echo "Create the canonical split first with:" >&2
  echo "  sbatch --partition=stampede slurm/data_preprocessing/brats_partition.sbatch" >&2
  echo "Or set FL_BRATS_SPLIT_PATH to an existing split manifest." >&2
  exit 2
}

extra_args=(--split-path "$split_path")
if [[ "${FL_BRATS_OVERWRITE_PARTITIONS:-0}" == "1" ]]; then
  extra_args+=(--overwrite)
fi

submit_partition() {
  local name="$1"
  local config="$2"

  sbatch --parsable \
    --partition="$partition" \
    --job-name="flb-part-${name}" \
    --output="logs/data_preprocessing/${name}_partition_%j.out" \
    --error="logs/data_preprocessing/${name}_partition_%j.err" \
    "$repo_root/slurm/data_preprocessing/prepare_experiment_partition.sbatch" \
    "$config" "${extra_args[@]}"
}

uneven_job="$(submit_partition fedavg-uneven configs/fedavg_uneven/baseline.yaml)"
even_job="$(submit_partition fedavg-even configs/fedavg_even/baseline.yaml)"

echo "fedavg_uneven_partition_job=$uneven_job"
echo "fedavg_even_partition_job=$even_job"
echo "split_path=$split_path"
echo "Wait for both jobs to finish successfully before submitting training."
