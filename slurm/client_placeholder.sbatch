#!/usr/bin/env bash
# PLACEHOLDER ONLY: no client or training is started.
#SBATCH --job-name=fl-brats-client-placeholder
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
echo "Array task: ${SLURM_ARRAY_TASK_ID}; hostname: $(hostname)"
echo "Future command: scripts/run_client_placeholder.sh --client-id ${SLURM_ARRAY_TASK_ID} --server-address HOST --partition-manifest PATH"
