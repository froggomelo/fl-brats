#!/usr/bin/env bash
#SBATCH --job-name=fl-brats-env
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm-env-%j.out
set -euo pipefail
echo "job=${SLURM_JOB_ID:-} node=${SLURMD_NODENAME:-} partition=${SLURM_JOB_PARTITION:-}"
apptainer exec --cleanenv "$APPTAINER_IMAGE" fedbrats info
apptainer exec --cleanenv "$APPTAINER_IMAGE" fedbrats check-environment
