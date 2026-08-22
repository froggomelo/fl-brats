#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG [--split-path PATH] [--overwrite]" >&2
  exit 2
fi
exec fedbrats prepare-experiment-partition --config "$@"
