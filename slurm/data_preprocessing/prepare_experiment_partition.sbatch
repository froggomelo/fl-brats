#!/usr/bin/env bash
#SBATCH --job-name=flb-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: sbatch $0 CONFIG" >&2; exit 2; }

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$repo_root"

if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

source "$repo_root/apptainer/select_image.sh"
fl_brats_select_apptainer_image "$repo_root"
[[ -f "$APPTAINER_IMAGE" ]] || {
  echo "Apptainer image missing: $APPTAINER_IMAGE" >&2
  exit 2
}

data_root="${FL_BRATS_DATA_ROOT:-/datasets/${USER}/brats}"
[[ -d "$data_root" ]] || {
  echo "BraTS data root missing: $data_root" >&2
  exit 2
}
mpl_config_dir="${FL_BRATS_CACHE_ROOT:-$repo_root/cache}/matplotlib"
mkdir -p "$mpl_config_dir"

echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "image=$APPTAINER_IMAGE"
echo "data_root=$data_root"

exec apptainer exec --cleanenv \
  --env "PYTHONPATH=/workspace/src" \
  --env "MPLCONFIGDIR=$mpl_config_dir" \
  --env "FL_BRATS_DATA_ROOT=$data_root" \
  --env "FL_BRATS_RAW_ROOT=${FL_BRATS_RAW_ROOT:?set FL_BRATS_RAW_ROOT in .env}" \
  --bind "$repo_root:$repo_root" \
  --bind "$repo_root/src:/workspace/src:ro" \
  --bind "$data_root:$data_root:ro" \
  --pwd "$repo_root" \
  "$APPTAINER_IMAGE" \
  python -m fl_brats prepare-experiment-partition --config "$@"
