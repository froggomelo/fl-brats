#!/usr/bin/env bash
set -euo pipefail

interval="${SQUEUE_INTERVAL:-5}"

if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue is not available on PATH" >&2
  exit 127
fi

while true; do
  clear
  date
  if [[ "$#" -gt 0 ]]; then
    squeue "$@"
  else
    squeue --me
  fi
  sleep "$interval"
done
