#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${1:?usage: $0 CONFIG CLIENT_ID SERVER_ADDRESS [extra arguments]}"
CLIENT_ID="${2:?usage: $0 CONFIG CLIENT_ID SERVER_ADDRESS [extra arguments]}"
ADDRESS="${3:?usage: $0 CONFIG CLIENT_ID SERVER_ADDRESS [extra arguments]}"
shift 3
cd "$ROOT"
echo "fedbrats start-federated-client --config $CONFIG --client-id $CLIENT_ID --server-address $ADDRESS $*"
exec fedbrats start-federated-client --config "$CONFIG" --client-id "$CLIENT_ID" --server-address "$ADDRESS" "$@"
