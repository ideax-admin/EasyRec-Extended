"""Abstract base class for ranking engines."""
import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)


class BaseRankingEngine(ABC):
    """Abstract base class for all ranking engines.

    Subclasses must implement the :meth:`rank` method, which scores and
    orders a list of candidate items for a given user context.
    """

    @abstractmethod
    def rank(self, items: List, user_context, features: dict = None) -> List:
        """Score and sort candidate items for the given user context.

        Args:
            items: List of :class:`core.models.Item` candidates.
            user_context: A :class:`core.models.UserContext` instance.
            features: Optional pre-assembled feature dict.

        Returns:
            Sorted list of :class:`core.models.Item` objects (highest score first).
        """
        raise NotImplementedError
