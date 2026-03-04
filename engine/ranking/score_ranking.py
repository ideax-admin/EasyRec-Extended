"""Score-based ranking engine (default, no model required)."""
import logging
from typing import List

from engine.ranking.base_ranking import BaseRankingEngine

logger = logging.getLogger(__name__)


class ScoreRankingEngine(BaseRankingEngine):
    """Ranks items by their existing ``score`` attribute (descending).

    This is the default ranking engine used when no EasyRec model is loaded.
    It preserves the original :class:`engine.recommendation_engine.RecommendationEngine`
    behaviour of sorting by score.
    """

    def rank(self, items: List, user_context=None, features: dict = None) -> List:
        """Sort items by score in descending order.

        Args:
            items: List of :class:`core.models.Item` candidates.
            user_context: Ignored by this engine; kept for interface parity.
            features: Ignored by this engine; kept for interface parity.

        Returns:
            Items sorted by ``score`` attribute (highest first).
        """
        ranked = sorted(items, key=lambda x: getattr(x, 'score', 0), reverse=True)
        logger.debug(f"ScoreRankingEngine: ranked {len(ranked)} items by score")
        return ranked
