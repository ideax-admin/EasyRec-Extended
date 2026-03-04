"""Embedding-based recall engine using EasyRec model inference and Faiss ANN."""
import logging
from typing import List, Optional

import numpy as np

from engine.recall.base_recall import BaseRecallEngine

logger = logging.getLogger(__name__)


class EmbeddingRecallEngine(BaseRecallEngine):
    """Recall engine that computes user embeddings via an EasyRec model and
    retrieves candidate items via an ANN index (e.g. :class:`FaissANNIndex`).

    Uses :class:`easyrec_extended.adapters.model_inference.EasyRecModelInference`
    (or a :class:`easyrec_extended.model_manager.ModelManager`) to generate a
    user embedding vector, then queries the supplied *ann_index* for the
    nearest-neighbour item IDs.

    When *ann_index* is *None* or the index is empty, the engine falls back to
    :class:`engine.recall.fallback_recall.FallbackRecallEngine`.

    Args:
        model_inference: An :class:`EasyRecModelInference` instance used to
            compute embeddings.  When *None*, ``model_manager`` is used as a
            fallback.
        feature_service: An optional :class:`FeatureService` used to fetch
            user features before calling the model.
        model_manager: An optional :class:`ModelManager` used for inference
            when ``model_inference`` is not provided.
        ann_index: An optional :class:`~easyrec_extended.ann.base_index.BaseANNIndex`
            instance used for nearest-neighbour retrieval.
    """

    def __init__(
        self,
        model_inference=None,
        feature_service=None,
        model_manager=None,
        ann_index=None,
    ):
        """Initialize the embedding recall engine.

        Args:
            model_inference: EasyRecModelInference instance (optional).
            feature_service: FeatureService instance (optional).
            model_manager: ModelManager instance used when model_inference
                is None (optional).
            ann_index: BaseANNIndex instance for ANN retrieval (optional).
        """
        self._model_inference = model_inference
        self._feature_service = feature_service
        self._model_manager = model_manager
        self._ann_index = ann_index

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

            embedding = result.get('user_embedding')
            if embedding is None:
                embedding = result.get('output_0')
            if embedding is not None:
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            return None
        except Exception as e:
            logger.error(f"EmbeddingRecallEngine: embedding computation failed: {e}")
            return None

    def recall(self, request) -> List:
        """Retrieve candidate items via embedding-based ANN search.

        1. Computes a user embedding via the configured model.
        2. If an ANN index is loaded, queries it for the nearest *candidate_size*
           item IDs and returns them as :class:`core.models.Item` objects.
        3. Falls back to :class:`~engine.recall.fallback_recall.FallbackRecallEngine`
           when no embedding can be produced or no ANN index is available.

        Args:
            request: A :class:`core.models.RecommendationRequest` instance.

        Returns:
            List of :class:`core.models.Item` objects.
        """
        from engine.recall.fallback_recall import FallbackRecallEngine

        user_id = getattr(getattr(request, 'user_context', None), 'user_id', None)
        embedding = self.get_user_embedding(user_id) if user_id else None

        if embedding is None:
            logger.debug("EmbeddingRecallEngine: no embedding; falling back to FallbackRecallEngine")
            return FallbackRecallEngine().recall(request)

        if self._ann_index is None or self._ann_index.size == 0:
            logger.debug(
                "EmbeddingRecallEngine: ANN index not available or empty; "
                "falling back to FallbackRecallEngine"
            )
            return FallbackRecallEngine().recall(request)

        candidate_size = getattr(request, 'candidate_size', 100)
        try:
            query = np.array(embedding, dtype=np.float32)
            hits = self._ann_index.search(query, top_k=candidate_size)
        except Exception as exc:
            logger.error("EmbeddingRecallEngine: ANN search failed: %s", exc)
            return FallbackRecallEngine().recall(request)

        from core.models import Item, ItemType, RecommendationSource

        items = []
        for rank, (item_id, distance) in enumerate(hits):
            item = Item(
                item_id=item_id,
                title=f"Item {item_id}",
                category="unknown",
                item_type=ItemType.PRODUCT,
                score=1.0 / (1.0 + distance),
                source=RecommendationSource.RECALL,
            )
            items.append(item)
        logger.debug("EmbeddingRecallEngine: ANN search returned %d items", len(items))
        return items

    def build_index(self, item_embeddings: np.ndarray, item_ids: List[str]) -> None:
        """Build the ANN index from item embeddings.

        Args:
            item_embeddings: 2-D float32 array of shape ``(n, dim)``.
            item_ids: List of string item IDs with length *n*.
        """
        if self._ann_index is None:
            raise RuntimeError(
                "No ANN index configured. Pass an ann_index to the constructor."
            )
        self._ann_index.build(item_embeddings, item_ids)
        logger.info("EmbeddingRecallEngine: built ANN index with %d items", self._ann_index.size)

    def load_index(self, index_path: str) -> None:
        """Load a pre-built ANN index from disk.

        Args:
            index_path: Base path (without extension) of the saved index.
        """
        if self._ann_index is None:
            raise RuntimeError(
                "No ANN index configured. Pass an ann_index to the constructor."
            )
        self._ann_index.load(index_path)
        logger.info(
            "EmbeddingRecallEngine: loaded ANN index from %s (%d items)",
            index_path,
            self._ann_index.size,
        )
