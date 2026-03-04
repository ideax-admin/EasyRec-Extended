"""Embedding-based recall engine using EasyRec model inference."""
import logging
from typing import List, Optional

from engine.recall.base_recall import BaseRecallEngine

logger = logging.getLogger(__name__)


class EmbeddingRecallEngine(BaseRecallEngine):
    """Recall engine that computes user embeddings via an EasyRec model.

    Uses :class:`easyrec_extended.adapters.model_inference.EasyRecModelInference`
    (or a :class:`easyrec_extended.model_manager.ModelManager`) to generate a
    user embedding vector, which can then be used for approximate
    nearest-neighbour (ANN) retrieval against an item index.

    ANN retrieval (e.g. via Faiss or Milvus) is not yet wired in; the
    :meth:`recall` method returns an empty list with a TODO marker until the
    index integration is complete.

    Args:
        model_inference: An :class:`EasyRecModelInference` instance used to
            compute embeddings.  When *None*, ``model_manager`` is used as a
            fallback.
        feature_service: An optional :class:`FeatureService` used to fetch
            user features before calling the model.
        model_manager: An optional :class:`ModelManager` used for inference
            when ``model_inference`` is not provided.
    """

    def __init__(self, model_inference=None, feature_service=None, model_manager=None):
        """Initialize the embedding recall engine.

        Args:
            model_inference: EasyRecModelInference instance (optional).
            feature_service: FeatureService instance (optional).
            model_manager: ModelManager instance used when model_inference
                is None (optional).
        """
        self._model_inference = model_inference
        self._feature_service = feature_service
        self._model_manager = model_manager

    def get_user_embedding(self, user_id: str, user_features: Optional[dict] = None) -> Optional[list]:
        """Compute a user embedding using the configured model.

        Args:
            user_id: The user identifier.
            user_features: Pre-fetched user feature dict.  When *None* the
                feature service (if available) is queried automatically.

        Returns:
            Embedding as a list of floats, or *None* if the model is not
            loaded or the call fails.
        """
        # Determine the active inference backend
        inference = self._model_inference
        if inference is None and self._model_manager is not None:
            # Use model_manager.predict directly
            if not self._model_manager.is_ready():
                logger.warning("EmbeddingRecallEngine: model_manager not ready")
                return None
        elif inference is not None and not inference.is_loaded:
            logger.warning("EmbeddingRecallEngine: no model loaded, cannot compute embedding")
            return None
        elif inference is None:
            logger.warning("EmbeddingRecallEngine: no model or model_manager provided")
            return None

        if user_features is None and self._feature_service is not None:
            user_features = self._feature_service.get_user_features(user_id)

        features = user_features or {}
        try:
            if self._model_inference is not None:
                result = self._model_inference.predict(features)
            else:
                result = self._model_manager.predict(features)

            embedding = result.get('user_embedding') or result.get('output_0')
            if embedding is not None:
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            return None
        except Exception as e:
            logger.error(f"EmbeddingRecallEngine: embedding computation failed: {e}")
            return None

    def recall(self, request) -> List:
        """Retrieve candidate items via embedding-based ANN search.

        Currently returns an empty list — ANN index integration (Faiss/Milvus)
        is not yet implemented.

        Args:
            request: A :class:`core.models.RecommendationRequest` instance.

        Returns:
            Empty list (ANN retrieval TODO).
        """
        user_id = getattr(getattr(request, 'user_context', None), 'user_id', None)
        embedding = self.get_user_embedding(user_id) if user_id else None

        if embedding is None:
            logger.debug("EmbeddingRecallEngine: no embedding; returning empty recall")
            return []

        # TODO: perform ANN index lookup (Faiss / Milvus) using ``embedding``
        logger.debug(
            "EmbeddingRecallEngine: ANN retrieval not yet implemented; "
            "returning empty list"
        )
        return []
