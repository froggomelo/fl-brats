#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$root/scripts/launch_experiment.py" "$root/configs/comm_eff_uneven/best_p_0_1.yaml" "$@"
