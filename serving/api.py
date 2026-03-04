"""
REST API routes for EasyRec-Extended serving layer.

Endpoints:
  POST /api/v1/recommend               - Get personalized recommendations
  GET  /api/v1/models                  - List loaded model versions
  POST /api/v1/models/reload           - Trigger hot model reload
  GET  /api/v1/experiments             - List A/B experiments
  POST /api/v1/experiments             - Create A/B experiment
  GET  /api/v1/experiments/<name>/results - Get experiment results
"""
import logging
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Lazy references set by register_components()
_server = None
_model_manager = None
_experiment_manager = None


def register_components(server=None, model_manager=None, experiment_manager=None):
    """Bind serving components to this blueprint."""
    global _server, _model_manager, _experiment_manager
    _server = server
    _model_manager = model_manager
    _experiment_manager = experiment_manager


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


# ---------------------------------------------------------------------------
# Experiment endpoints
# ---------------------------------------------------------------------------

@api_bp.route('/experiments', methods=['GET'])
def list_experiments():
    """
    List all A/B experiments.

    Returns:
      JSON with a list of experiment config dicts.
    """
    if _experiment_manager is None:
        return jsonify({'experiments': []})
    return jsonify({'experiments': _experiment_manager.list_experiments()})


@api_bp.route('/experiments', methods=['POST'])
def create_experiment():
    """
    Create a new A/B experiment.

    Request body (JSON):
      name               (str, required)
      control_version    (str, required)
      treatment_version  (str, required)
      traffic_split      (float, optional, default=0.5)

    Returns:
      JSON with the created experiment config.
    """
    if _experiment_manager is None:
        return jsonify({'status': 'error', 'error': 'experiment manager not initialised'}), 503

    data = request.get_json(silent=True) or {}
    name = data.get('name')
    control = data.get('control_version')
    treatment = data.get('treatment_version')

    if not name or not control or not treatment:
        return jsonify({
            'status': 'error',
            'error': 'name, control_version, and treatment_version are required',
        }), 400

    try:
        config = _experiment_manager.create_experiment(
            name=name,
            control_version=control,
            treatment_version=treatment,
            traffic_split=float(data.get('traffic_split', 0.5)),
        )
        return jsonify({'status': 'success', 'experiment': config}), 201
    except ValueError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 409
    except Exception as e:
        logger.error(f"Create experiment failed: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500


@api_bp.route('/experiments/<name>/results', methods=['GET'])
def get_experiment_results(name: str):
    """
    Get summary statistics for an A/B experiment.

    Returns:
      JSON with control and treatment metric summaries.
    """
    if _experiment_manager is None:
        return jsonify({'status': 'error', 'error': 'experiment manager not initialised'}), 503

    try:
        results = _experiment_manager.get_experiment_results(name)
        return jsonify({'experiment': name, 'results': results})
    except ValueError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Get experiment results failed: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500
