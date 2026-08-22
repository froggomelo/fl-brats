#!/usr/bin/env bash
set -euo pipefail
fedbrats create-partition --split-path "$FL_BRATS_DATA_ROOT/partitions/fixed_quantity_skew/split_manifest.json" --output-dir "$FL_BRATS_DATA_ROOT/partitions/fixed_quantity_skew"
