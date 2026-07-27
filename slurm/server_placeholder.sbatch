#!/usr/bin/env bash
# PLACEHOLDER ONLY: no server is started.
#SBATCH --job-name=fl-brats-server-placeholder
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
set -euo pipefail
echo "Allocated hostname: $(hostname)"
echo "Future command: scripts/run_server_placeholder.sh --server-address $(hostname) --server-port 8080"
