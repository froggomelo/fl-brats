#!/usr/bin/env bash
# Complete BraTS 2021 preprocessing pipeline. Submit with:
#   sbatch --export=ALL,FL_BRATS_DATA_ROOT=/datasets/$USER/fedbrats,FL_BRATS_IMAGE=/path/fl-brats.sif slurm/brats_pipeline.sbatch
#SBATCH --job-name=brats-pipeline
#SBATCH --partition=bigbatch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --output=logs/brats_pipeline_%j.out
#SBATCH --error=logs/brats_pipeline_%j.err

set -euo pipefail

: "${FL_BRATS_DATA_ROOT:?Set FL_BRATS_DATA_ROOT, e.g. /datasets/$USER/fedbrats}"
: "${FL_BRATS_IMAGE:?Set FL_BRATS_IMAGE to the Apptainer SIF image}"
: "${FL_BRATS_RAW_ROOT:=/datasets/${USER}/brats/raw/brats2021/PKG - RSNA-ASNR-MICCAI-BraTS-2021/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_TrainingSet}"

raw_host="$FL_BRATS_RAW_ROOT"
processed_host="$FL_BRATS_DATA_ROOT/processed"
partitions_host="$FL_BRATS_DATA_ROOT/partitions"
log_host="${FL_BRATS_LOG_ROOT:-$PWD/logs}"
container_root="/datasets/${USER}/fedbrats"

[[ -d "$raw_host" ]] || { echo "Missing raw training dataset: $raw_host" >&2; exit 1; }
mkdir -p "$processed_host/brats2021_v1" "$partitions_host/fixed_quantity_skew" "$log_host"

echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-unknown} node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "started=$(date --iso-8601=seconds)"

# Raw data are bound read-only; all generated artefacts are separately writable.
runner=(apptainer exec --cleanenv
  --bind "$raw_host:$container_root/raw/brats2021:ro"
  --bind "$processed_host:$container_root/processed:rw"
  --bind "$partitions_host:$container_root/partitions:rw"
  --bind "$log_host:/logs:rw"
  "$FL_BRATS_IMAGE" fedbrats)

run() { echo "+ fedbrats $*"; "${runner[@]}" "$@"; }
on_signal() { echo "Termination received; completed subjects remain resumable." >&2; exit 143; }
trap on_signal TERM INT

raw="$container_root/raw/brats2021"
processed="$container_root/processed/brats2021_v1"
partitions="$container_root/partitions/fixed_quantity_skew"

# 1. Strict validation gates all later artefacts.
run validate-data --root "$raw" --strict --report-path "$processed/validation_report.json"

# 2. Index immutable raw data.
run index-data --root "$raw" --output-dir "$processed"

# 3. Confirm manifest-only operation; this creates metadata but no subject volumes.
run preprocess-data --index-path "$processed/dataset_index.csv" --raw-root "$raw" \
  --output-dir "$processed" preprocessing=brats2021_v1 preprocessing.materialize=false

# 4. Materialize reusable volumes atomically and resume safely after requeueing.
run preprocess-data --index-path "$processed/dataset_index.csv" --raw-root "$raw" \
  --output-dir "$processed" --resume preprocessing=brats2021_v1 preprocessing.materialize=true

# 5. Freeze split and fixed quantity-skew partition, then deep-verify references.
run create-split --index-path "$processed/dataset_index.csv" \
  --manifest-path "$partitions/split_manifest.json"
run create-partition --split-path "$partitions/split_manifest.json" --output-dir "$partitions"
run verify-manifests --partition-root "$partitions" --check-nifti

echo "completed=$(date --iso-8601=seconds)"
