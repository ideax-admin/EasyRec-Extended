"""Tests for the FaissANNIndex implementation."""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 64
N = 100


def _random_embeddings(n: int = N, dim: int = DIM, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, dim), dtype=np.float32)


def _item_ids(n: int = N) -> list:
    return [f"item_{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Tests that require faiss
# ---------------------------------------------------------------------------

try:
    import faiss  # noqa: F401
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


@unittest.skipUnless(_FAISS_AVAILABLE, "faiss not installed")
class TestFaissANNIndexFlat(unittest.TestCase):
    """Tests using the Flat (exact) index type."""

    def setUp(self):
        from easyrec_extended.ann.faiss_index import FaissANNIndex
        self.index = FaissANNIndex(dimension=DIM, index_type='Flat')
        self.embeddings = _random_embeddings()
        self.ids = _item_ids()

    def test_size_before_build(self):
        self.assertEqual(self.index.size, 0)

    def test_build_sets_size(self):
        self.index.build(self.embeddings, self.ids)
        self.assertEqual(self.index.size, N)

    def test_search_returns_top_k(self):
        self.index.build(self.embeddings, self.ids)
        query = self.embeddings[0]
        results = self.index.search(query, top_k=10)
        self.assertEqual(len(results), 10)

    def test_search_result_format(self):
        self.index.build(self.embeddings, self.ids)
        query = self.embeddings[0]
        results = self.index.search(query, top_k=5)
        for item_id, dist in results:
            self.assertIsInstance(item_id, str)
            self.assertIsInstance(dist, float)
            self.assertGreaterEqual(dist, 0.0)

    def test_search_nearest_is_self(self):
        """The nearest neighbour of a vector should be itself."""
        self.index.build(self.embeddings, self.ids)
        query = self.embeddings[7]
        results = self.index.search(query, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 'item_7')

    def test_search_top_k_capped_by_size(self):
        """Requesting more results than indexed vectors should not error."""
        self.index.build(self.embeddings, self.ids)
        results = self.index.search(self.embeddings[0], top_k=N + 50)
        self.assertEqual(len(results), N)

    def test_add_increases_size(self):
        self.index.build(self.embeddings, self.ids)
        extra = _random_embeddings(n=20, seed=99)
        extra_ids = [f"extra_{i}" for i in range(20)]
        self.index.add(extra, extra_ids)
        self.assertEqual(self.index.size, N + 20)

    def test_save_and_load(self):
        self.index.build(self.embeddings, self.ids)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_index')
            self.index.save(path)
            self.assertTrue(os.path.exists(path + '.faiss'))
            self.assertTrue(os.path.exists(path + '.meta'))

            from easyrec_extended.ann.faiss_index import FaissANNIndex
            loaded = FaissANNIndex(dimension=DIM, index_type='Flat')
            loaded.load(path)
            self.assertEqual(loaded.size, N)

            # Results should be identical after reload
            q = self.embeddings[3]
            original_results = self.index.search(q, top_k=5)
            reloaded_results = loaded.search(q, top_k=5)
            self.assertEqual(
                [r[0] for r in original_results],
                [r[0] for r in reloaded_results],
            )

    def test_build_raises_on_empty(self):
        with self.assertRaises(ValueError):
            self.index.build(np.empty((0, DIM), dtype=np.float32), [])

    def test_add_before_build_raises(self):
        with self.assertRaises(RuntimeError):
            self.index.add(self.embeddings[:5], self.ids[:5])

    def test_search_before_build_raises(self):
        with self.assertRaises(RuntimeError):
            self.index.search(self.embeddings[0], top_k=5)

    def test_save_before_build_raises(self):
        with self.assertRaises(RuntimeError):
            self.index.save('/tmp/noop')


@unittest.skipUnless(_FAISS_AVAILABLE, "faiss not installed")
class TestFaissANNIndexIVF(unittest.TestCase):
    """Tests using the IVFFlat index type."""

    def test_build_and_search(self):
        from easyrec_extended.ann.faiss_index import FaissANNIndex
        embeddings = _random_embeddings(n=200)
        ids = _item_ids(n=200)
        index = FaissANNIndex(dimension=DIM, index_type='IVFFlat', nlist=10)
        index.build(embeddings, ids)
        self.assertEqual(index.size, 200)
        results = index.search(embeddings[0], top_k=5)
        self.assertEqual(len(results), 5)

    def test_nlist_auto_reduced_when_too_large(self):
        """nlist should be silently reduced when fewer vectors than nlist."""
        from easyrec_extended.ann.faiss_index import FaissANNIndex
        embeddings = _random_embeddings(n=20)
        ids = _item_ids(n=20)
        index = FaissANNIndex(dimension=DIM, index_type='IVFFlat', nlist=100)
        # Should not raise
        index.build(embeddings, ids)
        self.assertEqual(index.size, 20)


# ---------------------------------------------------------------------------
# Test graceful degradation when faiss is absent
# ---------------------------------------------------------------------------

class TestFaissANNIndexNoFaiss(unittest.TestCase):
    """Verify that importing FaissANNIndex without faiss raises ImportError."""

    def test_import_error_without_faiss(self):
        with patch.dict('sys.modules', {'faiss': None}):
            # Re-import the module with faiss patched out
            import importlib
            import easyrec_extended.ann.faiss_index as mod
            importlib.reload(mod)
            with self.assertRaises(ImportError):
                mod.FaissANNIndex(dimension=DIM)
            # Reload original state
            importlib.reload(mod)


if __name__ == '__main__':
    unittest.main()
