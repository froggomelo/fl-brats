#!/usr/bin/env bash
# Generic infrastructure-only SLURM template; configure partition externally.
#SBATCH --job-name=fl-brats-infra
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
apptainer exec --cleanenv "$APPTAINER_IMAGE" fedbrats validate-config
