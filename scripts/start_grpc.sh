#!/usr/bin/env bash
# Start the EasyRec-Extended gRPC server.
#
# Environment variables:
#   GRPC_PORT          - TCP port (default: 50051)
#   GRPC_MAX_WORKERS   - Thread pool size (default: 10)
#   RANKING_MODEL_PATH - Path to ranking SavedModel directory (optional)
#   RECALL_MODEL_PATH  - Path to recall SavedModel directory (optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

GRPC_PORT="${GRPC_PORT:-50051}"
GRPC_MAX_WORKERS="${GRPC_MAX_WORKERS:-10}"

echo "Starting gRPC server on port ${GRPC_PORT} with ${GRPC_MAX_WORKERS} workers..."

exec python - << PYEOF
import os
import logging
from core.config import get_config
from easyrec_extended.model_manager import ModelManager
from online.serving import RecommendationServer
from serving.grpc_service import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("start_grpc")

config = get_config()
model_manager = ModelManager()

ranking_path = os.environ.get("RANKING_MODEL_PATH")
recall_path = os.environ.get("RECALL_MODEL_PATH")
if ranking_path:
    try:
        model_manager.load_version("ranking", ranking_path)
    except Exception as e:
        logger.warning("Failed to load ranking model: %s", e)
if recall_path:
    try:
        model_manager.load_version("recall", recall_path)
    except Exception as e:
        logger.warning("Failed to load recall model: %s", e)

server = RecommendationServer(config)

serve(
    port=int(os.environ.get("GRPC_PORT", 50051)),
    max_workers=int(os.environ.get("GRPC_MAX_WORKERS", 10)),
    server=server,
    model_manager=model_manager,
    config=config,
)
PYEOF
