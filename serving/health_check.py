"""
Health check utilities for EasyRec-Extended.

Checks:
  - Model loaded status
  - Redis connectivity
  - Overall service health
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def check_model(model_manager=None) -> Dict[str, Any]:
    """
    Check whether a model is loaded and ready.

    Args:
        model_manager: A ModelManager instance, or None.

    Returns:
        Dict with 'status' ('ok' or 'unavailable') and optional detail.
    """
    if model_manager is None:
        return {'status': 'unavailable', 'detail': 'model manager not configured'}
    if model_manager.is_ready():
        return {
            'status': 'ok',
            'active_version': model_manager.active_version,
            'loaded_versions': model_manager.loaded_versions,
        }
    return {'status': 'unavailable', 'detail': 'no model version loaded'}


def check_redis(redis_client=None) -> Dict[str, Any]:
    """
    Check Redis connectivity by issuing a PING command.

    Args:
        redis_client: A redis.Redis instance, or None.

    Returns:
        Dict with 'status' ('ok' or 'unavailable') and optional detail.
    """
    if redis_client is None:
        return {'status': 'unavailable', 'detail': 'redis client not configured'}
    try:
        redis_client.ping()
        return {'status': 'ok'}
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return {'status': 'unavailable', 'detail': str(e)}


def health_check(
    model_manager=None,
    redis_client=None,
    service_name: str = 'EasyRec-Extended',
    version: str = '0.1.0',
) -> Dict[str, Any]:
    """
    Aggregate health check for the serving service.

    Args:
        model_manager: ModelManager instance (optional).
        redis_client: Redis client instance (optional).
        service_name: Human-readable service name.
        version: Service version string.

    Returns:
        Dict with overall 'status' ('healthy' or 'degraded') and per-component
        health details.
    """
    components = {
        'model': check_model(model_manager),
        'redis': check_redis(redis_client),
    }

    all_ok = all(c['status'] == 'ok' for c in components.values())
    overall = 'healthy' if all_ok else 'degraded'

    return {
        'status': overall,
        'service': service_name,
        'version': version,
        'components': components,
    }
