"""Faiss-backed ANN index implementation."""
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from easyrec_extended.ann.base_index import BaseANNIndex

logger = logging.getLogger(__name__)

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]
    _FAISS_AVAILABLE = False
    logger.warning(
        "faiss is not installed; FaissANNIndex will be unavailable. "
        "Install it with: pip install faiss-cpu"
    )


class FaissANNIndex(BaseANNIndex):
    """ANN index backed by Facebook's Faiss library.

    Supports two index types:

    * ``'Flat'`` — exact brute-force L2 search (no training required).
    * ``'IVFFlat'`` — inverted-file index with flat quantiser (requires
      training; faster for large-scale retrieval).

    String item IDs are mapped to sequential integer IDs internally and
    stored in ``idx_to_id`` / ``id_to_idx`` dictionaries.

    Args:
        dimension: Dimensionality of the embedding vectors.
        index_type: One of ``'Flat'`` or ``'IVFFlat'``.  Defaults to
            ``'IVFFlat'``.
        nlist: Number of Voronoi cells for the IVF index.  Ignored for
            ``'Flat'``.  Defaults to 100.
        nprobe: Number of cells to visit at search time.  Ignored for
            ``'Flat'``.  Defaults to 10.
        use_gpu: If *True* and a CUDA-enabled Faiss build is available, the
            index is moved to the first GPU.  Defaults to *False*.
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = 'IVFFlat',
        nlist: int = 100,
        nprobe: int = 10,
        use_gpu: bool = False,
    ) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required for FaissANNIndex. "
                "Install it with: pip install faiss-cpu"
            )
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu

        self._index: Optional[faiss.Index] = None  # type: ignore[name-defined]
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._next_idx: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_base_index(self) -> "faiss.Index":  # type: ignore[name-defined]
        """Return an untrained Faiss index for the configured *index_type*."""
        if self.index_type == 'Flat':
            return faiss.IndexFlatL2(self.dimension)
        # IVFFlat (default)
        quantiser = faiss.IndexFlatL2(self.dimension)
        return faiss.IndexIVFFlat(quantiser, self.dimension, self.nlist)

    def _wrap_with_id_map(self, base_index: "faiss.Index") -> "faiss.IndexIDMap":  # type: ignore[name-defined]
        return faiss.IndexIDMap(base_index)

    def _to_float32(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return np.ascontiguousarray(arr)

    def _assign_ids(self, ids: List[str]) -> np.ndarray:
        """Allocate sequential integer IDs for *ids* and update mappings."""
        int_ids = []
        for sid in ids:
            if sid not in self._id_to_idx:
                self._id_to_idx[sid] = self._next_idx
                self._idx_to_id[self._next_idx] = sid
                self._next_idx += 1
            int_ids.append(self._id_to_idx[sid])
        return np.array(int_ids, dtype=np.int64)

    # ------------------------------------------------------------------
    # BaseANNIndex interface
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Build the Faiss index from *embeddings* and *ids*.

        For IVF-based indexes the index is trained before vectors are added.

        Args:
            embeddings: 2-D float32 array of shape ``(n, dim)``.
            ids: List of string item IDs with length *n*.
        """
        if len(ids) == 0:
            raise ValueError("Cannot build an index with zero vectors.")

        vecs = self._to_float32(embeddings)
        base = self._create_base_index()

        # Train if the index requires it (IVF family)
        if hasattr(base, 'is_trained') and not base.is_trained:
            nlist_eff = min(self.nlist, len(ids))
            if nlist_eff != self.nlist:
                logger.warning(
                    "nlist (%d) > number of training vectors (%d); "
                    "reducing nlist to %d",
                    self.nlist, len(ids), nlist_eff,
                )
                base.nlist = nlist_eff
            base.train(vecs)

        # Wrap with IDMap so we can store arbitrary int64 IDs
        self._index = self._wrap_with_id_map(base)
        if hasattr(self._index.index, 'nprobe'):
            self._index.index.nprobe = self.nprobe

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            except Exception as exc:
                logger.warning("Failed to move index to GPU: %s", exc)

        int_ids = self._assign_ids(ids)
        self._index.add_with_ids(vecs, int_ids)
        logger.info("FaissANNIndex: built index with %d vectors (dim=%d)", len(ids), self.dimension)

    def add(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Incrementally add vectors to an existing index.

        Args:
            embeddings: 2-D float32 array of shape ``(n, dim)``.
            ids: List of string item IDs with length *n*.
        """
        if self._index is None:
            raise RuntimeError("Index has not been built yet. Call build() first.")

        vecs = self._to_float32(embeddings)
        int_ids = self._assign_ids(ids)
        self._index.add_with_ids(vecs, int_ids)
        logger.debug("FaissANNIndex: added %d vectors; total=%d", len(ids), self.size)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Retrieve the *top_k* most similar items by L2 distance.

        Args:
            query_embedding: 1-D float32 array of shape ``(dim,)``.
            top_k: Number of results to return.

        Returns:
            List of ``(item_id, distance)`` tuples, ordered by ascending
            distance.
        """
        if self._index is None:
            raise RuntimeError("Index has not been built yet. Call build() first.")

        query = self._to_float32(query_embedding)
        k = min(top_k, self.size)
        distances, indices = self._index.search(query, k)

        results: List[Tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            item_id = self._idx_to_id.get(int(idx))
            if item_id is not None:
                results.append((item_id, float(dist)))
        return results

    def save(self, path: str) -> None:
        """Save the Faiss index and ID mappings to *path*.

        Two files are written: ``<path>.faiss`` (the Faiss index binary) and
        ``<path>.meta`` (a pickle of the ID-mapping dictionaries).

        Args:
            path: Base path (without extension).
        """
        if self._index is None:
            raise RuntimeError("Index has not been built yet. Call build() first.")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Move back to CPU before serialising if on GPU
        index_to_save = self._index
        try:
            index_to_save = faiss.index_gpu_to_cpu(self._index)
        except Exception:
            pass

        faiss.write_index(index_to_save, path + '.faiss')

        meta = {
            'id_to_idx': self._id_to_idx,
            'idx_to_id': self._idx_to_id,
            'next_idx': self._next_idx,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
        }
        with open(path + '.meta', 'wb') as fh:
            pickle.dump(meta, fh)
        logger.info("FaissANNIndex: saved to %s(.faiss/.meta)", path)

    def load(self, path: str) -> None:
        """Load a previously saved index from *path*.

        Args:
            path: Base path (without extension) used during :meth:`save`.
        """
        index = faiss.read_index(path + '.faiss')

        with open(path + '.meta', 'rb') as fh:
            meta = pickle.load(fh)

        self._id_to_idx = meta['id_to_idx']
        self._idx_to_id = meta['idx_to_id']
        self._next_idx = meta['next_idx']
        self.dimension = meta['dimension']
        self.index_type = meta['index_type']
        self.nlist = meta.get('nlist', self.nlist)
        self.nprobe = meta.get('nprobe', self.nprobe)

        if hasattr(index, 'index') and hasattr(index.index, 'nprobe'):
            index.index.nprobe = self.nprobe

        self._index = index
        logger.info("FaissANNIndex: loaded from %s(.faiss/.meta), size=%d", path, self.size)

    @property
    def size(self) -> int:
        """Number of vectors stored in the index."""
        if self._index is None:
            return 0
        return int(self._index.ntotal)
