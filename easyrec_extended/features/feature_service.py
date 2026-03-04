"""
Feature Service - retrieves and assembles features for EasyRec inference.

Provides methods to fetch user/item features from a feature store (Redis
or any dict-backed store) and assemble them into the flat dict format
expected by EasyRec's serving signature.
"""
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _observe_feature_latency(feature_type: str, elapsed: float):
    """Record feature retrieval latency; silently ignores missing prometheus_client."""
    try:
        from easyrec_extended.metrics.prometheus_metrics import feature_service_latency_seconds
        feature_service_latency_seconds.labels(feature_type=feature_type).observe(elapsed)
    except Exception:
        pass


class FeatureService:
    """Retrieves user and item features and builds EasyRec input dicts."""

    def __init__(self, feature_store=None):
        """
        Args:
            feature_store: An optional backing store object that supports
                ``get(key)`` and ``mget(keys)`` methods (e.g. a Redis client
                or a simple dict wrapper).  When None, an empty in-memory
                dict is used as a fallback.
        """
        self._store = feature_store or {}

    def get_user_features(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve user features from the feature store.

        Args:
            user_id: The user identifier.

        Returns:
            Dict of feature name → value for the given user.
            Returns an empty dict if no features are found.
        """
        t0 = time.time()
        try:
            if hasattr(self._store, 'hgetall'):
                raw = self._store.hgetall(f"user:{user_id}")
                return {k.decode(): v.decode() for k, v in raw.items()} if raw else {}
            key = f"user:{user_id}"
            return self._store.get(key, {}) if isinstance(self._store, dict) else {}
        except Exception as e:
            logger.warning(f"Failed to fetch user features for {user_id}: {e}")
            return {}
        finally:
            _observe_feature_latency("user", time.time() - t0)

    def get_item_features(self, item_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Batch-retrieve item features from the feature store.

        Args:
            item_ids: List of item identifiers.

        Returns:
            Dict mapping item_id → feature dict.
            Items with no features are mapped to an empty dict.
        """
        t0 = time.time()
        result: Dict[str, Dict[str, Any]] = {}
        for item_id in item_ids:
            try:
                if hasattr(self._store, 'hgetall'):
                    raw = self._store.hgetall(f"item:{item_id}")
                    result[item_id] = (
                        {k.decode(): v.decode() for k, v in raw.items()} if raw else {}
                    )
                else:
                    key = f"item:{item_id}"
                    result[item_id] = (
                        self._store.get(key, {}) if isinstance(self._store, dict) else {}
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch item features for {item_id}: {e}")
                result[item_id] = {}
        _observe_feature_latency("item", time.time() - t0)
        return result

    def build_easyrec_features(
        self,
        user_features: Dict[str, Any],
        item_features: Dict[str, Any],
        context_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Assemble a flat feature dict suitable for EasyRec's serving input.

        EasyRec's serving signature expects a single flat dict where keys
        correspond to input field names defined in the pipeline config.

        Args:
            user_features: Dict of user-level features.
            item_features: Dict of item-level features for a *single* item.
            context_features: Optional dict of request-level context features.

        Returns:
            Merged flat feature dict.
        """
        merged: Dict[str, Any] = {}
        merged.update(user_features)
        merged.update(item_features)
        if context_features:
            merged.update(context_features)
        return merged
