#!/bin/bash
set -euo pipefail

python -m services.signaling_self_hosted.server &
SIG_PID=$!

python -m services.edge_gateway.app --config configs/edge_gateway.self_hosted.yaml &
EDGE_PID=$!

cleanup() {
  kill $EDGE_PID 2>/dev/null || true
  kill $SIG_PID 2>/dev/null || true
}

trap cleanup EXIT
wait
