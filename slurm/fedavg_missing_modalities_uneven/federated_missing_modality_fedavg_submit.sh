#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$repo_root/.env"
  set +a
fi

data_root="${FL_BRATS_DATA_ROOT:-/datasets/${USER}/brats}"
partition_root="${PARTITION_ROOT:-${data_root}/partitions/missing_modality_arbitrary_025}"
initial_weights_path="${INITIAL_WEIGHTS_PATH:-artifacts/initial_weights/fedamm_seed_${FL_BRATS_SEED:-42}.pt}"
for required in "$partition_root/client_0.json" "$partition_root/manifest.json" "$partition_root/split_manifest.json"; do
  [[ -f "$required" ]] || {
    echo "Required missing-modality partition input missing: $required" >&2
    echo "Create it first with: PARTITION_ROOT=\"$partition_root\" bash \"$repo_root/slurm/data_preprocessing/prepare_missing_modality_partition.sh\"" >&2
    exit 2
  }
done

export PARTITION_ROOT="$partition_root"
export INITIAL_WEIGHTS_PATH="$initial_weights_path"
export FEDERATED_RUN_ID="${FEDERATED_RUN_ID:-missing_modality_fedavg_$(date +%Y%m%d_%H%M%S)}"
export FEDERATED_EXPERIMENT_NAME="federated_missing_modality_fedavg"
export FEDERATED_MODEL="fedamm"
export FEDERATED_STRATEGY="fedavg"
# Exhaustive robustness evaluation runs the validation set once for each of the
# 15 non-empty modality combinations.  Doing that on every Flower round can
# exceed the round deadline before the server receives any client result.
# Keep round-level validation on the partition's natural missingness; run the
# exhaustive matrix separately from the resulting best/final checkpoint.
export FEDERATED_OVERRIDES="missing_modalities.enabled=true missing_modalities.mode=arbitrary missing_modalities.evaluation.evaluate_all_non_empty_combinations=false model.prototype_loss_weight=0.0 model.prototype_distillation_enabled=false model.global_prototype_alignment_enabled=false model.emit_prototypes=false model.modality_weighted_aggregation=false"

bash "$repo_root/slurm/fedavg_uneven/federated_baseline_submit.sh"
