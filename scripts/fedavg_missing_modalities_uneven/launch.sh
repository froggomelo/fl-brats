#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$root/scripts/launch_experiment.py" "$root/configs/fedavg_missing_modalities_uneven/alpha_0_1.yaml" "$@"
