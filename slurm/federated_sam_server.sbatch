#!/usr/bin/env bash
#SBATCH --job-name=fedbrats-sam-server
#SBATCH --output=logs/sam_server_%j.out
set -euo pipefail
: "${SAM_MODEL:?SAM_MODEL must be sam1 or sam2}"
# Bind the configured pretrained directory into Apptainer; downloads stay disabled.
fedbrats verify-pretrained-model --model-config "configs/model/${SAM_MODEL}.yaml"
fedbrats "train-federated-${SAM_MODEL}"
