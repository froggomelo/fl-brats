#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

partition="${PARTITION:-bigbatch}"
image="${IMAGE:-apptainer/fl-brats-batch.sif}"
fedamm_weights="artifacts/initial_weights/fedamm_seed_42.pt"

if [[ ! -s "$fedamm_weights" ]]; then
  mkdir -p "$(dirname "$fedamm_weights")"
  "$root/slurm/_shared/run_with_configured_environment.sh" true "$image" \
    fedbrats create-initial-weights \
    --model-config configs/model/fedamm.yaml \
    --output-dir artifacts/initial_weights \
    --seed 42
fi

dependency=""
submit_smoke() {
  local config="$1"
  local result
  local -a command=(
    python3 scripts/launch_experiment.py "$config"
    --mode slurm --partition "$partition" --image "$image"
  )
  if [[ -n "$dependency" ]]; then
    command+=(--afterany "$dependency")
  fi
  result="$("${command[@]}")"
  printf '%s\n' "$result"
  dependency="$(python3 -c \
    'import json,sys; print(json.load(sys.stdin)[0]["jobs"]["cleanup"])' \
    <<<"$result")"
}

submit_smoke configs/smoke/missing_modalities.yaml
submit_smoke configs/smoke/fedmpq.yaml
submit_smoke configs/smoke/selective_transmission.yaml
