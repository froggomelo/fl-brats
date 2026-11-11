#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# The first argument is the config unless it is a launcher option.
config="$root/configs/fedmpq_uneven/homogeneous_4bit.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then config="$1"; shift; fi
exec python "$root/scripts/launch_experiment.py" "$config" "$@"
