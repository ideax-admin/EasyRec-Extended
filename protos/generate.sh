#!/usr/bin/env bash
# Generate Python gRPC stubs from protos/recommendation.proto
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python -m grpc_tools.protoc \
  -I "$SCRIPT_DIR" \
  --python_out="$REPO_ROOT/serving" \
  --grpc_python_out="$REPO_ROOT/serving" \
  "$SCRIPT_DIR/recommendation.proto"

echo "Stubs written to $REPO_ROOT/serving/"
