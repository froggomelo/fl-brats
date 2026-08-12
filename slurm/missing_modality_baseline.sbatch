#!/usr/bin/env bash
# Arbitrary example:
# sbatch --export=ALL,MM_MODE=arbitrary,MM_T1CE_PROBABILITY=0.4 slurm/missing_modality_baseline.sbatch
# Complete-policy example:
# sbatch --export=ALL,MM_MODE=complete,MM_CLIENT_POLICY='{client_0: [t1,t1ce,t2,flair], client_1: [t1,t2,flair], client_2: [t2,flair]}' slurm/missing_modality_baseline.sbatch
#SBATCH --job-name=brats-missing-modality
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/missing_modality_%j.out
#SBATCH --error=logs/slurm/missing_modality_%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

source "$repo_root/apptainer/select_image.sh"
fl_brats_select_apptainer_image "$repo_root"
[[ -f "$APPTAINER_IMAGE" ]] || { echo "Apptainer image missing: $APPTAINER_IMAGE" >&2; exit 2; }
data_root="${FL_BRATS_DATA_ROOT:-/datasets/${USER}/brats}"
output_root="${FL_BRATS_OUTPUT_ROOT:-outputs}"
log_root="${FL_BRATS_LOG_ROOT:-logs}"
cache_root="${FL_BRATS_CACHE_ROOT:-cache}"
checkpoint_root="${FL_BRATS_CHECKPOINT_ROOT:-checkpoints}"
partition_root="${PARTITION_ROOT:-$data_root/partitions/missing_modality}"
index_path="${DATASET_INDEX_PATH:-$data_root/processed/brats2021_v1/dataset_index.csv}"
weights_path="${INITIAL_WEIGHTS_PATH:-artifacts/initial_weights/unet_seed_42.pt}"
mode="${MM_MODE:-arbitrary}"

mkdir -p "$partition_root" "$output_root" "$log_root/slurm" "$cache_root" "$checkpoint_root"
[[ -f "$index_path" && -f "$weights_path" ]] || { echo "Index or initial weights missing" >&2; exit 2; }

run() {
  echo "partition=${SLURM_JOB_PARTITION:-unknown} image=$APPTAINER_IMAGE"
  apptainer exec --nv --cleanenv \
    --bind "$data_root:$data_root:ro" --bind "$partition_root:$partition_root" \
    --bind "$output_root:$output_root" --bind "$log_root:$log_root" \
    --bind "$cache_root:$cache_root" --bind "$checkpoint_root:$checkpoint_root" \
    "$APPTAINER_IMAGE" fedbrats "$@"
}

run create-split --index-path "$index_path" --manifest-path "$partition_root/split_manifest.json" --seed "${FL_BRATS_SEED:-42}"

overrides=("missing_modalities.enabled=true" "missing_modalities.mode=$mode" "missing_modalities.seed=${FL_BRATS_SEED:-42}")
if [[ "$mode" == "arbitrary" || "$mode" == "mixed" ]]; then
  overrides+=("missing_modalities.arbitrary.per_modality_missing_probability.t1ce=${MM_T1CE_PROBABILITY:-0.0}")
fi
if [[ "$mode" == "complete" || "$mode" == "mixed" ]] && [[ -n "${MM_CLIENT_POLICY:-}" ]]; then
  overrides+=("missing_modalities.complete.client_available_modalities=$MM_CLIENT_POLICY")
fi

run create-partition --split-path "$partition_root/split_manifest.json" --output-dir "$partition_root" "${overrides[@]}"
run verify-manifests --partition-root "$partition_root"
run inspect-modalities --partition-root "$partition_root"
run train-centralized --index-path "$index_path" --split-path "$partition_root/split_manifest.json" --partition-path "$partition_root/manifest.json" --initial-weights "$weights_path" --output-root "$output_root/missing_modality" --epochs "${EPOCHS:-50}" --batch-size "${BATCH_SIZE:-1}" --workers "${CENTRALIZED_WORKERS:-0}" --no-pin-memory --seed "${FL_BRATS_SEED:-42}" --device cuda
