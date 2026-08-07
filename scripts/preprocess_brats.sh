#!/usr/bin/env bash
set -euo pipefail
fedbrats preprocess-data --index-path "$FL_BRATS_DATA_ROOT/processed/brats2021_v1/dataset_index.csv" --raw-root "${FL_BRATS_RAW_ROOT:-$FL_BRATS_DATA_ROOT/raw/brats2021}" --output-dir "$FL_BRATS_DATA_ROOT/processed/brats2021_v1" --resume
