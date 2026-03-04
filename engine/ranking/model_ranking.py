"""Model-based ranking engine using EasyRec inference for scoring."""
import logging
from typing import List, Optional

from engine.ranking.base_ranking import BaseRankingEngine

logger = logging.getLogger(__name__)


class ModelRankingEngine(BaseRankingEngine):
    """Ranks items using a live EasyRec model for score prediction.

    For each candidate item the engine:

    1. Fetches user features via :class:`FeatureService`.
    2. Fetches item features via :class:`FeatureService`.
    3. Assembles a flat feature dict using
       :meth:`FeatureService.build_easyrec_features`.
    4. Calls :meth:`ModelManager.predict` (or falls back to
       :meth:`EasyRecModelInference.predict`) to obtain a score.
    5. Assigns the score back to the item and sorts the list.

    Batch prediction is used when the model manager supports it, for
    efficiency when processing many candidates.

    Args:
        model_manager: A :class:`easyrec_extended.model_manager.ModelManager`
            instance used for inference.
        feature_service: A :class:`easyrec_extended.features.feature_service.FeatureService`
            instance used to build the input feature dict.
        score_key: The key to extract from the model output dict as the item
            score.  Defaults to ``'score'``; also tries ``'output_0'``.
    """

    def __init__(self, model_manager=None, feature_service=None, score_key: str = 'score'):
        """Initialise the model ranking engine.

        Args:
            model_manager: ModelManager instance (optional).
            feature_service: FeatureService instance (optional).
            score_key: Key in the model output to use as ranking score.
        """
        self._model_manager = model_manager
        self._feature_service = feature_service
        self._score_key = score_key

    def _extract_score(self, prediction: dict) -> float:
        """Extract a scalar score from a model prediction dict.

        Tries ``score_key`` first, then ``output_0`` as a fallback.

        Args:
            prediction: Raw prediction dict from the model.

        Returns:
            Float score, or 0.0 if no recognisable key is found.
        """
        for key in (self._score_key, 'output_0', 'proba', 'logit'):
            if key in prediction:
                val = prediction[key]
                try:
                    # val may be a numpy scalar or array
                    return float(val.flat[0]) if hasattr(val, 'flat') else float(val)
                except (TypeError, ValueError):
                    pass
        return 0.0

    def rank(self, items: List, user_context=None, features: dict = None) -> List:
        """Score and rank items using the EasyRec model.

        Falls back to score-attribute sorting when the model manager is not
        ready or no feature service is configured.

        Args:
            items: List of :class:`core.models.Item` candidates.
            user_context: A :class:`core.models.UserContext` instance.
            features: Optional pre-assembled user feature dict.  When *None*
                the feature service is queried automatically.

        Returns:
            Items sorted by model-predicted score (highest first).
        """
        if not items:
            return items

        if self._model_manager is None or not self._model_manager.is_ready():
            logger.warning("ModelRankingEngine: model not ready, falling back to score sort")
            return sorted(items, key=lambda x: getattr(x, 'score', 0), reverse=True)

        user_id = getattr(user_context, 'user_id', None) if user_context else None
        user_feats = features or {}
        if not user_feats and self._feature_service is not None and user_id:
            user_feats = self._feature_service.get_user_features(user_id)

        item_ids = [item.item_id for item in items]
        item_feats_map: dict = {}
        if self._feature_service is not None:
            item_feats_map = self._feature_service.get_item_features(item_ids)

        # Build feature batch for all items
        feature_batch = []
        for item in items:
            item_feats = item_feats_map.get(item.item_id, {})
            merged = self._feature_service.build_easyrec_features(
                user_feats, item_feats
            ) if self._feature_service is not None else {**user_feats, **item_feats}
            feature_batch.append(merged)

        # Batch predict
        try:
            predictions = self._model_manager.batch_predict(feature_batch)
        except Exception as e:
            logger.error(f"ModelRankingEngine: batch_predict failed: {e}")
            predictions = [{} for _ in items]

        # Assign scores and sort
        for item, pred in zip(items, predictions):
            item.score = self._extract_score(pred) if pred else getattr(item, 'score', 0.0)

        ranked = sorted(items, key=lambda x: getattr(x, 'score', 0), reverse=True)
        logger.debug(f"ModelRankingEngine: ranked {len(ranked)} items")
        return ranked
