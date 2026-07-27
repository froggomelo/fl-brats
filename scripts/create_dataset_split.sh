#!/usr/bin/env bash
set -euo pipefail
fedbrats create-split --index-path "$FL_BRATS_DATA_ROOT/processed/brats2021_v1/dataset_index.csv" --manifest-path "$FL_BRATS_DATA_ROOT/partitions/fixed_quantity_skew/split_manifest.json"
