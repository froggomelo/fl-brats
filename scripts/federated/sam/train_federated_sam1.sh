#!/usr/bin/env bash
set -euo pipefail
fedbrats train-federated-sam1 model=sam1 experiment=federated_sam1 "$@"
