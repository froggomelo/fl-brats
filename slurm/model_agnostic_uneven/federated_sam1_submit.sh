#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs/model_agnostic_uneven
sbatch --export=ALL,SAM_MODEL=sam1 slurm/model_agnostic_uneven/federated_sam_server.sbatch
sbatch --export=ALL,SAM_MODEL=sam1 slurm/model_agnostic_uneven/federated_sam_clients.sbatch
