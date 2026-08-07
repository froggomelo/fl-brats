#!/usr/bin/env bash
set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

data_root="${FL_BRATS_DATA_ROOT:-/datasets/${USER}/brats}"
partition_root="${PARTITION_ROOT:-${data_root}/partitions/missing_modality_arbitrary_025}"
PARTITION_ROOT="$partition_root" bash slurm/prepare_missing_modality_partition.sh

export PARTITION_ROOT="$partition_root"
export FEDERATED_RUN_ID="${FEDERATED_RUN_ID:-missing_modality_fedavg_$(date +%Y%m%d_%H%M%S)}"
export FEDERATED_EXPERIMENT_NAME="federated_missing_modality_fedavg"
export FEDERATED_MODEL="fedamm"
export FEDERATED_STRATEGY="fedavg"
export FEDERATED_OVERRIDES="missing_modalities.enabled=true missing_modalities.mode=arbitrary model.prototype_loss_weight=0.0 model.prototype_distillation_enabled=false model.global_prototype_alignment_enabled=false model.emit_prototypes=false model.modality_weighted_aggregation=false"

bash slurm/federated_baseline_submit.sh
