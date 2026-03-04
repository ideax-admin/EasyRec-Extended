"""Tests for EmbeddingRecallEngine with ANN index integration."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.recall.embedding_recall import EmbeddingRecallEngine
from core.models import Item, RecommendationRequest, UserContext


def _make_request(user_id: str = 'u1', candidate_size: int = 10) -> RecommendationRequest:
    ctx = UserContext(user_id=user_id, session_id='s1', timestamp=datetime.now())
    return RecommendationRequest(user_context=ctx, candidate_size=candidate_size)


class TestEmbeddingRecallWithANNIndex(unittest.TestCase):
    """Tests for EmbeddingRecallEngine when an ANN index is available."""

    def _make_ann_index(self, hits=None):
        """Return a mock ANN index."""
        if hits is None:
            hits = [('item_0', 0.1), ('item_1', 0.2), ('item_2', 0.3)]
        ann = MagicMock()
        ann.size = 100
        ann.search.return_value = hits
        return ann

    def _make_model_inference(self, embedding=None):
        mi = MagicMock()
        mi.is_loaded = True
        mi.predict.return_value = {
            'user_embedding': np.array(embedding or [0.1] * 64, dtype=np.float32)
        }
        return mi

    def test_recall_returns_items_from_ann_index(self):
        ann = self._make_ann_index([('item_A', 0.05), ('item_B', 0.15)])
        mi = self._make_model_inference()
        engine = EmbeddingRecallEngine(model_inference=mi, ann_index=ann)
        items = engine.recall(_make_request())
        self.assertEqual(len(items), 2)
        item_ids = [it.item_id for it in items]
        self.assertIn('item_A', item_ids)
        self.assertIn('item_B', item_ids)

    def test_recall_items_are_Item_instances(self):
        ann = self._make_ann_index()
        mi = self._make_model_inference()
        engine = EmbeddingRecallEngine(model_inference=mi, ann_index=ann)
        items = engine.recall(_make_request())
        for item in items:
            self.assertIsInstance(item, Item)

    def test_recall_score_based_on_distance(self):
        """Closer (smaller distance) items should have higher scores."""
        ann = self._make_ann_index([('item_close', 0.0), ('item_far', 10.0)])
        mi = self._make_model_inference()
        engine = EmbeddingRecallEngine(model_inference=mi, ann_index=ann)
        items = engine.recall(_make_request())
        self.assertEqual(items[0].item_id, 'item_close')
        self.assertGreater(items[0].score, items[1].score)

    def test_ann_search_called_with_candidate_size(self):
        ann = self._make_ann_index([])
        mi = self._make_model_inference()
        engine = EmbeddingRecallEngine(model_inference=mi, ann_index=ann)
        engine.recall(_make_request(candidate_size=50))
        ann.search.assert_called_once()
        _, kwargs = ann.search.call_args
        self.assertEqual(kwargs.get('top_k') or ann.search.call_args[0][1], 50)


class TestEmbeddingRecallFallback(unittest.TestCase):
    """Tests for EmbeddingRecallEngine fallback behaviour."""

    def test_fallback_when_no_ann_index(self):
        """Without an ANN index the engine falls back to FallbackRecallEngine."""
        mi = MagicMock()
        mi.is_loaded = True
        mi.predict.return_value = {'user_embedding': np.array([0.1] * 64)}
        engine = EmbeddingRecallEngine(model_inference=mi, ann_index=None)
        items = engine.recall(_make_request(candidate_size=5))
        # FallbackRecallEngine generates synthetic items
        self.assertGreater(len(items), 0)
        for item in items:
            self.assertIsInstance(item, Item)

    def test_fallback_when_ann_index_empty(self):
        ann = MagicMock()
        ann.size = 0
        mi = MagicMock()
        mi.is_loaded = True
        mi.predict.return_value = {'user_embedding': np.array([0.1] * 64)}
        engine = EmbeddingRecallEngine(model_inference=mi, ann_index=ann)
        items = engine.recall(_make_request(candidate_size=5))
        self.assertGreater(len(items), 0)

    def test_fallback_when_no_user_id(self):
        """No user_id → no embedding → fallback."""
        ann = MagicMock()
        ann.size = 10
        engine = EmbeddingRecallEngine(ann_index=ann)
        ctx = UserContext(user_id='', session_id='s', timestamp=datetime.now())
        request = RecommendationRequest(user_context=ctx, candidate_size=5)
        # Patching user_context.user_id to be falsy
        request.user_context.user_id = ''
        items = engine.recall(request)
        self.assertIsInstance(items, list)

    def test_fallback_when_ann_search_raises(self):
        """ANN search exception should trigger fallback."""
        ann = MagicMock()
        ann.size = 100
        ann.search.side_effect = RuntimeError("index error")
        mi = MagicMock()
        mi.is_loaded = True
        mi.predict.return_value = {'user_embedding': np.array([0.1] * 64)}
        engine = EmbeddingRecallEngine(model_inference=mi, ann_index=ann)
        items = engine.recall(_make_request())
        self.assertGreater(len(items), 0)


class TestEmbeddingRecallBuildLoadIndex(unittest.TestCase):
    """Tests for build_index and load_index helper methods."""

    def test_build_index_delegates_to_ann(self):
        ann = MagicMock()
        engine = EmbeddingRecallEngine(ann_index=ann)
        emb = np.random.rand(10, 64).astype(np.float32)
        ids = [f"i_{x}" for x in range(10)]
        engine.build_index(emb, ids)
        ann.build.assert_called_once_with(emb, ids)

    def test_load_index_delegates_to_ann(self):
        ann = MagicMock()
        engine = EmbeddingRecallEngine(ann_index=ann)
        engine.load_index('/tmp/test_index')
        ann.load.assert_called_once_with('/tmp/test_index')

    def test_build_index_raises_without_ann(self):
        engine = EmbeddingRecallEngine()
        with self.assertRaises(RuntimeError):
            engine.build_index(np.zeros((5, 64)), ['a'] * 5)

    def test_load_index_raises_without_ann(self):
        engine = EmbeddingRecallEngine()
        with self.assertRaises(RuntimeError):
            engine.load_index('/tmp/noop')


if __name__ == '__main__':
    unittest.main()
