#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 CONFIG RUN_DIR" >&2; exit 2; }
config="$1"
run_dir="$2"
mkdir -p "$run_dir"
host="$(hostname -f 2>/dev/null || hostname)"
port="$(python - "$config" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1], encoding="utf-8")).get("federated", {}).get("server_port", 8080))
PY
)"
address="$host:$port"
printf '%s\n' "$address" > "$run_dir/server_address.txt"
exec "$(dirname "$0")/run_with_configured_environment.sh" "$config" \
  fedbrats start-federated-server --config "$config" --run-dir "$run_dir" --server-address "$address"
