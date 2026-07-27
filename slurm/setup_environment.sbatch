#!/usr/bin/env bash
# Submit once per checkout: sbatch slurm/setup_environment.sbatch
#SBATCH --job-name=fedbrats-setup
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/setup_%j.out
#SBATCH --error=logs/slurm/setup_%j.err

set -euo pipefail

trap 'echo "Environment setup interrupted" >&2; exit 143' TERM INT

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
cache_dir="${UV_CACHE_DIR:-$repo_root/.uv-cache}"
marker_dir="${FL_BRATS_SETUP_MARKER_DIR:-$repo_root/.setup}"
mkdir -p "$cache_dir" "$marker_dir" logs/slurm

echo "hostname=$(hostname) job=${SLURM_JOB_ID:-unknown}"
echo "repository=$repo_root"
export UV_CACHE_DIR="$cache_dir"

# bootstrap_cluster installs uv if missing, validates Python 3.11, and performs
# a locked dependency synchronization. This runs only on the allocated node.
bash scripts/bootstrap_cluster.sh
uv run fedbrats validate-config
uv run fedbrats check-environment || true

printf 'completed_at=%s\njob_id=%s\n' "$(date --iso-8601=seconds)" "${SLURM_JOB_ID:-unknown}" \
  > "$marker_dir/environment_ready"
echo "Environment setup complete: $marker_dir/environment_ready"
