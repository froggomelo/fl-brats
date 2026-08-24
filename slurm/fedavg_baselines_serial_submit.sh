#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

uneven_result="$(bash "$script_dir/fedavg_uneven/federated_baseline_submit.sh")"
printf '%s\n' "$uneven_result"
last_cleanup="$(python3 -c 'import json, sys; print(json.load(sys.stdin)[-1]["jobs"]["cleanup"])' <<<"$uneven_result")"

exec bash "$script_dir/fedavg_even/federated_baseline_submit.sh" --afterany "$last_cleanup"
