"""Fallback recall engine using synthetic items when no model is available."""
import logging
from typing import List

from engine.recall.base_recall import BaseRecallEngine

logger = logging.getLogger(__name__)


class FallbackRecallEngine(BaseRecallEngine):
    """Generates synthetic candidate items when no EasyRec model is loaded.

    This engine is the default fallback and preserves the original behaviour
    of :class:`engine.recommendation_engine.RecommendationEngine` when no
    real recall source is configured.
    """

    def recall(self, request) -> List:
        """Return a list of synthetic :class:`core.models.Item` objects.

        Args:
            request: A :class:`core.models.RecommendationRequest` instance.

        Returns:
            List of synthetic :class:`core.models.Item` objects with
            incrementing scores.
        """
        from core.models import Item, ItemType, RecommendationSource

        candidate_size = getattr(request, 'candidate_size', 100)
        items = []
        for i in range(candidate_size):
            item = Item(
                item_id=f"item_{i}",
                title=f"Product {i}",
                category="electronics",
                item_type=ItemType.PRODUCT,
                score=0.5 + (i % 50) / 100,
                source=RecommendationSource.RECALL,
            )
            items.append(item)
        logger.debug(f"FallbackRecallEngine generated {len(items)} synthetic items")
        return items
