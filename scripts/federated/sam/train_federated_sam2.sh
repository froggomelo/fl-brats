#!/usr/bin/env bash
set -euo pipefail
fedbrats train-federated-sam2 model=sam2 experiment=federated_sam2 "$@"
