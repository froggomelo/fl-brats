#!/usr/bin/env bash
#SBATCH --job-name=brats-preprocess
#SBATCH --partition=${FL_BRATS_CPU_PARTITION:-cpu}
#SBATCH --cpus-per-task=${FL_BRATS_CPUS:-4}
#SBATCH --mem=${FL_BRATS_MEM:-32G}
#SBATCH --output=logs/brats_preprocess_%j.out
set -euo pipefail
trap 'echo "termination received; completed subjects remain resumable"; exit 143' TERM INT
: "${FL_BRATS_DATA_ROOT:?set FL_BRATS_DATA_ROOT}"
apptainer exec --bind "$FL_BRATS_DATA_ROOT/raw:/datasets/${USER}/fedbrats/raw:ro" --bind "$FL_BRATS_DATA_ROOT/processed:/datasets/${USER}/fedbrats/processed:rw" "${FL_BRATS_IMAGE:?set FL_BRATS_IMAGE}" fedbrats preprocess-data --index-path "/datasets/${USER}/fedbrats/processed/brats2021_v1/dataset_index.csv" --raw-root "/datasets/${USER}/fedbrats/raw/brats2021" --output-dir "/datasets/${USER}/fedbrats/processed/brats2021_v1" --resume
