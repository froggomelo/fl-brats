#!/usr/bin/env bash
set -Eeuo pipefail

# Continuously mirror cluster logs and run outputs to this machine.
# Defaults can be overridden with environment variables:
#   REMOTE_LOG_DIR=/path/on/cluster/logs \
#   REMOTE_OUTPUT_DIR=/path/on/cluster/outputs \
#   LOCAL_LOG_DIR=/path/here/logs \
#   LOCAL_OUTPUT_DIR=/path/here/outputs \
#   INTERVAL_SECONDS=600 \
#   ./scripts/sync_mscluster_logs.sh

REMOTE_HOST="${REMOTE_HOST:-mscluster}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-~/fl-brats/logs/}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-~/fl-brats/outputs/}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-./logs/}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-./outputs/}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
SYNC_OUTPUTS="${SYNC_OUTPUTS:-1}"

mkdir -p "$LOCAL_LOG_DIR" "$LOCAL_OUTPUT_DIR"

echo "Syncing ${REMOTE_HOST}:${REMOTE_LOG_DIR} -> ${LOCAL_LOG_DIR}"
if [[ "$SYNC_OUTPUTS" == "1" ]]; then
  echo "Syncing ${REMOTE_HOST}:${REMOTE_OUTPUT_DIR} -> ${LOCAL_OUTPUT_DIR}"
else
  echo "Output sync disabled with SYNC_OUTPUTS=${SYNC_OUTPUTS}"
fi
echo "Interval: ${INTERVAL_SECONDS}s"
echo "Press Ctrl+C to stop."

sync_path() {
  local label="$1"
  local remote_dir="$2"
  local local_dir="$3"
  shift 3

  if rsync -azvh --partial --info=stats2,progress2 "$@" \
    "${REMOTE_HOST}:${remote_dir%/}/" \
    "${local_dir%/}/"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label} sync finished"
  else
    status=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label} sync failed with exit code ${status}" >&2
  fi
}

while true; do
  started_at="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$started_at] sync started"

  sync_path "logs" "$REMOTE_LOG_DIR" "$LOCAL_LOG_DIR"

  if [[ "$SYNC_OUTPUTS" == "1" ]]; then
    sync_path "outputs" "$REMOTE_OUTPUT_DIR" "$LOCAL_OUTPUT_DIR" \
      --exclude='**/.hydra/**' \
      --exclude='**/__pycache__/**' \
      --exclude='**/*.tmp' \
      --exclude='**/*.part'
  fi

  sleep "$INTERVAL_SECONDS"
done
