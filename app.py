import logging
import os
import threading
from flask import Flask, jsonify, request
from online.serving import RecommendationServer
from core.config import get_config
from easyrec_extended.model_manager import ModelManager
from serving.api import api_bp, register_components
from serving.health_check import health_check as _health_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
config = get_config()
server = RecommendationServer(config)

# Initialise model manager and optionally pre-load model versions from env
model_manager = ModelManager()
_ranking_model_path = os.environ.get('RANKING_MODEL_PATH')
_recall_model_path = os.environ.get('RECALL_MODEL_PATH')
if _ranking_model_path:
    try:
        model_manager.load_version('ranking', _ranking_model_path)
        logger.info(f"Loaded ranking model from {_ranking_model_path}")
    except Exception as e:
        logger.warning(f"Failed to load ranking model: {e}")
if _recall_model_path:
    try:
        model_manager.load_version('recall', _recall_model_path)
        logger.info(f"Loaded recall model from {_recall_model_path}")
    except Exception as e:
        logger.warning(f"Failed to load recall model: {e}")

# Register serving components and blueprint
register_components(server=server, model_manager=model_manager)
app.register_blueprint(api_bp)

# gRPC configuration
_GRPC_ENABLED = os.environ.get('GRPC_ENABLED', 'false').lower() in ('1', 'true', 'yes')
_GRPC_PORT = int(os.environ.get('GRPC_PORT', '50051'))


@app.route('/recommend', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    result_size = request.args.get('result_size', type=int)
    result = server.get_recommendations(user_id=user_id, result_size=result_size)
    return jsonify(result)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(_health_check(model_manager=model_manager,
                                 service_name=config.SERVICE_NAME,
                                 version=config.VERSION))


@app.route('/', methods=['GET'])
def index():
    info = {
        'service': config.SERVICE_NAME,
        'version': config.VERSION,
        'endpoints': {
            'recommend': '/recommend?user_id=<user_id>',
            'health': '/health',
            'api': '/api/v1/',
        }
    }
    if _GRPC_ENABLED:
        info['grpc_port'] = _GRPC_PORT
    return jsonify(info)


def _start_grpc_server():
    """Start the gRPC server in a background thread."""
    from serving.grpc_service import serve_grpc
    logger.info(f"Starting gRPC server on port {_GRPC_PORT}")
    serve_grpc(port=_GRPC_PORT, server=server, model_manager=model_manager)


if __name__ == '__main__':
    logger.info(f'Starting {config.SERVICE_NAME} v{config.VERSION}')
    if _GRPC_ENABLED:
        grpc_thread = threading.Thread(target=_start_grpc_server, daemon=True)
        grpc_thread.start()
        logger.info(f"gRPC server thread started (port {_GRPC_PORT})")
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.DEBUG)