#!/usr/bin/env bash
#SBATCH --job-name=fedbrats-sam-client
#SBATCH --output=logs/sam_client_%j_%t.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
set -euo pipefail
: "${SAM_MODEL:?SAM_MODEL must be sam1 or sam2}"
fedbrats verify-pretrained-model --model-config "configs/model/${SAM_MODEL}.yaml"
fedbrats "train-federated-${SAM_MODEL}"
