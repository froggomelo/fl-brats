#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: $0 CONFIG SERVER_ADDRESS [extra arguments]}"
ADDRESS="${2:?usage: $0 CONFIG SERVER_ADDRESS [extra arguments]}"
shift 2
cd "$ROOT"
echo "fedbrats start-federated-server --config $CONFIG --server-address $ADDRESS $*"
exec fedbrats start-federated-server --config "$CONFIG" --server-address "$ADDRESS" "$@"
