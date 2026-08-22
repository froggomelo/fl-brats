#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec python "$root/scripts/launch_experiment.py" "$root/configs/model_agnostic_uneven/sam2/baseline.yaml" "$@"
