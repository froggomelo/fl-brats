#!/usr/bin/env bash
set -euo pipefail
fedbrats index-data --root "${FL_BRATS_DATA_ROOT:?set FL_BRATS_DATA_ROOT}/raw/brats2021" --output-dir "${FL_BRATS_DATA_ROOT}/processed/brats2021_v1"
