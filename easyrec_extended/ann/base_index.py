"""Abstract base class for ANN index implementations."""
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class BaseANNIndex(ABC):
    """Abstract base class for Approximate Nearest Neighbour indexes.

    Subclasses implement a specific backend (e.g. Faiss, ScaNN, Milvus) and
    must provide all five abstract methods below.
    """

    @abstractmethod
    def build(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Build the index from a batch of item embeddings.

        Args:
            embeddings: 2-D float32 array of shape ``(n, dim)``.
            ids: List of string item IDs, one per row in *embeddings*.
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Incrementally add vectors to an already-built index.

        Args:
            embeddings: 2-D float32 array of shape ``(n, dim)``.
            ids: List of string item IDs, one per row in *embeddings*.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Retrieve the *top_k* most similar items.

        Args:
            query_embedding: 1-D float32 array of shape ``(dim,)``.
            top_k: Number of results to return.

        Returns:
            List of ``(item_id, distance)`` tuples ordered by ascending
            distance (i.e. most similar first).
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk.

        Args:
            path: File-system path (without extension) where the index and
                the ID-mapping metadata will be written.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously saved index from disk.

        Args:
            path: File-system path (without extension) used when the index
                was originally saved.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors currently stored in the index."""
        raise NotImplementedError
