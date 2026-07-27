#!/usr/bin/env bash
#SBATCH --job-name=brats-validate
#SBATCH --partition=${FL_BRATS_CPU_PARTITION:-cpu}
#SBATCH --cpus-per-task=${FL_BRATS_CPUS:-4}
#SBATCH --mem=${FL_BRATS_MEM:-16G}
#SBATCH --output=logs/brats_validate_%j.out
set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"
: "${FL_BRATS_DATA_ROOT:?set FL_BRATS_DATA_ROOT}"
apptainer exec --bind "$FL_BRATS_DATA_ROOT:/datasets/${USER}/fedbrats:ro" "${FL_BRATS_IMAGE:?set FL_BRATS_IMAGE}" fedbrats validate-data --root "/datasets/${USER}/fedbrats/raw/brats2021" --strict
