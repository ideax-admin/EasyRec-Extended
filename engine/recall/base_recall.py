"""Abstract base class for recall engines."""
import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)


class BaseRecallEngine(ABC):
    """Abstract base class for all recall engines.

    Subclasses must implement the :meth:`recall` method, which retrieves
    candidate items for a given recommendation request.
    """

    @abstractmethod
    def recall(self, request) -> List:
        """Retrieve candidate items for the given request.

        Args:
            request: A :class:`core.models.RecommendationRequest` instance.

        Returns:
            List of :class:`core.models.Item` objects.
        """
        raise NotImplementedError
