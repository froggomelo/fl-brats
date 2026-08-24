#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

partition="${FL_BRATS_PARTITION:-${FL_BRATS_SLURM_PARTITION:-stampede}}"
split_path="${FL_BRATS_SPLIT_PATH:-data/splits/brats2021.json}"
mkdir -p logs/data_preprocessing

[[ -f "$split_path" ]] || {
  echo "Dataset split missing: $split_path" >&2
  echo "Set FL_BRATS_SPLIT_PATH to the existing BraTS split manifest." >&2
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
