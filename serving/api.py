"""
REST API routes for EasyRec-Extended serving layer.

Endpoints:
  POST /api/v1/recommend       - Get personalized recommendations
  GET  /api/v1/models          - List loaded model versions
  POST /api/v1/models/reload   - Trigger hot model reload
"""
import logging
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Lazy references set by register_components()
_server = None
_model_manager = None


def register_components(server=None, model_manager=None):
    """Bind serving components to this blueprint."""
    global _server, _model_manager
    _server = server
    _model_manager = model_manager


@api_bp.route('/recommend', methods=['POST'])
def recommend():
    """
    Get personalized recommendations.

    Request body (JSON):
      user_id       (str, required)
      result_size   (int, optional, default=20)
      candidate_size (int, optional, default=100)
      filters       (dict, optional)
      policies      (list, optional)
      business_rules (list, optional)

    Returns:
      JSON with status, request_id, user_id, items, processing_time_ms
    """
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'error': 'user_id is required'}), 400

    if _server is None:
        return jsonify({'status': 'error', 'error': 'serving backend not initialised'}), 503

    result = _server.get_recommendations(
        user_id=user_id,
        result_size=data.get('result_size'),
        candidate_size=data.get('candidate_size'),
        filters=data.get('filters'),
        policies=data.get('policies'),
        business_rules=data.get('business_rules'),
    )
    status_code = 200 if result.get('status') == 'success' else 500
    return jsonify(result), status_code


@api_bp.route('/models', methods=['GET'])
def list_models():
    """
    List all loaded model versions.

    Returns:
      JSON with loaded_versions and active_version.
    """
    if _model_manager is None:
        return jsonify({'loaded_versions': [], 'active_version': None})

    return jsonify({
        'loaded_versions': _model_manager.loaded_versions,
        'active_version': _model_manager.active_version,
        'is_ready': _model_manager.is_ready(),
    })


@api_bp.route('/models/reload', methods=['POST'])
def reload_model():
    """
    Trigger a hot model reload.

    Request body (JSON):
      version    (str, required) - version label to reload
      model_dir  (str, required) - path to new SavedModel directory

    Returns:
      JSON with status and message.
    """
    if _model_manager is None:
        return jsonify({'status': 'error', 'error': 'model manager not initialised'}), 503

    data = request.get_json(silent=True) or {}
    version = data.get('version')
    model_dir = data.get('model_dir')

    if not version or not model_dir:
        return jsonify({'status': 'error', 'error': 'version and model_dir are required'}), 400

    try:
        _model_manager.reload_version(version, model_dir)
        return jsonify({'status': 'success', 'message': f"Version '{version}' reloaded from {model_dir}"})
    except ValueError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Model reload failed: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500
