#!/usr/bin/env bash
set -euo pipefail
sbatch --export=ALL,SAM_MODEL=sam2 slurm/federated_sam_server.sbatch
sbatch --export=ALL,SAM_MODEL=sam2 slurm/federated_sam_clients.sbatch
