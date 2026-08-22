#!/usr/bin/env bash
#SBATCH --job-name=flb-partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: sbatch $0 CONFIG" >&2; exit 2; }
exec scripts/data_preprocessing/prepare_experiment_partition.sh "$@"
