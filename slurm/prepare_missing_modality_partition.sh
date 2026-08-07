#!/usr/bin/env bash
set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

: "${APPTAINER_IMAGE:=${FL_BRATS_IMAGE:-}}"
: "${APPTAINER_IMAGE:?Set APPTAINER_IMAGE or FL_BRATS_IMAGE to the built fl-brats image}"

data_root="${FL_BRATS_DATA_ROOT:-/datasets/${USER}/brats}"
output_root="${FL_BRATS_OUTPUT_ROOT:-outputs}"
log_root="${FL_BRATS_LOG_ROOT:-logs}"
cache_root="${FL_BRATS_CACHE_ROOT:-cache}"
checkpoint_root="${FL_BRATS_CHECKPOINT_ROOT:-checkpoints}"
partition_root="${PARTITION_ROOT:-${data_root}/partitions/missing_modality_arbitrary_025}"
index_path="${DATASET_INDEX_PATH:-${data_root}/processed/brats2021_v1/dataset_index.csv}"
mpl_config_dir="${MPLCONFIGDIR:-${cache_root}/matplotlib}"

mkdir -p "$partition_root" "$output_root" "$log_root/slurm" "$cache_root" "$checkpoint_root" "$mpl_config_dir"
[[ -f "$index_path" ]] || { echo "Dataset index missing: $index_path" >&2; exit 2; }

run_fedbrats() {
  apptainer exec --cleanenv \
    --env "MPLCONFIGDIR=$mpl_config_dir" \
    --bind "$repo_root/src:/workspace/src:ro" \
    --bind "$repo_root/configs:/workspace/configs:ro" \
    --bind "$data_root:$data_root:ro" \
    --bind "$partition_root:$partition_root" \
    --bind "$output_root:$output_root" \
    --bind "$log_root:$log_root" \
    --bind "$cache_root:$cache_root" \
    --bind "$checkpoint_root:$checkpoint_root" \
    "$APPTAINER_IMAGE" fedbrats "$@"
}

if [[ "${RECREATE_PARTITION:-1}" == "1" ]] || [[ ! -f "$partition_root/manifest.json" ]]; then
  run_fedbrats create-split \
    --index-path "$index_path" \
    --manifest-path "$partition_root/split_manifest.json" \
    --seed "${FL_BRATS_SEED:-42}"

  run_fedbrats create-partition \
    --split-path "$partition_root/split_manifest.json" \
    --output-dir "$partition_root" \
    missing_modalities.enabled=true \
    missing_modalities.mode=arbitrary \
    missing_modalities.seed="${FL_BRATS_SEED:-42}" \
    missing_modalities.arbitrary.per_modality_missing_probability.t1=0.25 \
    missing_modalities.arbitrary.per_modality_missing_probability.t1ce=0.25 \
    missing_modalities.arbitrary.per_modality_missing_probability.t2=0.25 \
    missing_modalities.arbitrary.per_modality_missing_probability.flair=0.25
else
  echo "Using existing missing-modality partition: $partition_root"
fi

run_fedbrats verify-manifests --partition-root "$partition_root"
run_fedbrats inspect-modalities --partition-root "$partition_root"
echo "partition_root=$partition_root"
