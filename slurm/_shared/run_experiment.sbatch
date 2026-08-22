#!/usr/bin/env bash
# Resources and per-run log paths are supplied from YAML by launch.py.
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$repo_root"
[[ $# -gt 1 ]] || { echo "Usage: $0 CONFIG COMMAND..." >&2; exit 2; }
config_path="$1"
shift
use_container="false"
image=""
if [[ -n "$config_path" ]]; then
  readarray -t settings < <(python - "$config_path" <<'PY'
import sys, yaml
value = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
slurm = value.get("slurm", {})
print(str(bool(slurm.get("use_apptainer", False))).lower())
print(slurm.get("image", ""))
PY
  )
  use_container="${settings[0]:-false}"
  image="${settings[1]:-}"
fi

echo "job_id=${SLURM_JOB_ID:-manual} host=$(hostname) command=$*"
if [[ "$use_container" == "true" ]]; then
  [[ -f "$image" ]] || { echo "Configured Apptainer image does not exist: $image" >&2; exit 2; }
  exec apptainer exec --nv --bind "$repo_root:$repo_root" --pwd "$repo_root" "$image" "$@"
fi
exec "$@"
