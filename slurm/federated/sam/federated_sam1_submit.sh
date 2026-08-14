#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs/slurm/federated/sam
sbatch --export=ALL,SAM_MODEL=sam1 slurm/federated/sam/federated_sam_server.sbatch
sbatch --export=ALL,SAM_MODEL=sam1 slurm/federated/sam/federated_sam_clients.sbatch
