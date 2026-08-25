#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
set -a
# shellcheck disable=SC1091
. "$repo_root/.env"
set +a
fedbrats validate-data --root "${FL_BRATS_RAW_ROOT:-${FL_BRATS_DATA_ROOT:?set FL_BRATS_DATA_ROOT}/raw/brats2021}" --strict --report-path "${FL_BRATS_DATA_ROOT}/processed/brats2021_v1/validation_report.json"
