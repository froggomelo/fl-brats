#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
config="${1:-$root/configs/fedamm_uneven/alpha_0_1.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
exec python "$root/scripts/launch_experiment.py" "$config" "$@"
