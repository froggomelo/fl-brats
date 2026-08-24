#!/usr/bin/env bash
set -euo pipefail
config="$1"
shift
repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$repo_root"
readarray -t settings < <(python - "$config" <<'PY'
import sys, yaml
slurm = yaml.safe_load(open(sys.argv[1], encoding="utf-8")).get("slurm", {})
print(str(bool(slurm.get("use_apptainer", False))).lower())
print(slurm.get("image", ""))
PY
)
if [[ "${settings[0]:-false}" == "true" ]]; then
  image="${settings[1]:-}"
  [[ -f "$image" ]] || { echo "Configured Apptainer image does not exist: $image" >&2; exit 2; }
  exec apptainer exec --nv --bind "$repo_root:$repo_root" --pwd "$repo_root" "$image" "$@"
fi
exec "$@"
