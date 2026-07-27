#!/usr/bin/env bash
#SBATCH --job-name=brats-partition
#SBATCH --partition=${FL_BRATS_CPU_PARTITION:-cpu}
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/brats_partition_%j.out
set -euo pipefail
: "${FL_BRATS_DATA_ROOT:?set FL_BRATS_DATA_ROOT}"
root="$FL_BRATS_DATA_ROOT"
fedbrats create-split --index-path "$root/processed/brats2021_v1/dataset_index.csv" --manifest-path "$root/partitions/fixed_quantity_skew/split_manifest.json"
fedbrats create-partition --split-path "$root/partitions/fixed_quantity_skew/split_manifest.json" --output-dir "$root/partitions/fixed_quantity_skew"
fedbrats verify-manifests --partition-root "$root/partitions/fixed_quantity_skew"
