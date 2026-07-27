#!/usr/bin/env bash
set -euo pipefail
fedbrats inspect-data --summary-path "$FL_BRATS_DATA_ROOT/processed/brats2021_v1/dataset_summary.json"
fedbrats inspect-split --manifest-path "$FL_BRATS_DATA_ROOT/partitions/fixed_quantity_skew/split_manifest.json"
fedbrats inspect-partition --manifest-path "$FL_BRATS_DATA_ROOT/partitions/fixed_quantity_skew/manifest.json"
