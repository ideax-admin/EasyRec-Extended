import unittest
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.recommendation_engine import RecommendationEngine
from core.models import RecommendationRequest, UserContext, RecommendationResult
from datetime import datetime
import uuid


class _MockRecallEngine:
    """Minimal recall engine stub that returns configurable items."""

    def __init__(self, items=None, delay_s=0.0, raise_exc=False):
        self.items = items or []
        self.delay_s = delay_s
        self.raise_exc = raise_exc
        self.call_count = 0

    def recall(self, request):
        self.call_count += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.raise_exc:
            raise RuntimeError("recall failed")
        return self.items


class TestRecommendationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RecommendationEngine()

    def _make_request(self, user_id='user_1', result_size=5, candidate_size=10):
        user_context = UserContext(
            user_id=user_id,
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        return RecommendationRequest(
            user_context=user_context,
            candidate_size=candidate_size,
            result_size=result_size,
        )

    def test_recommend_returns_result(self):
        result = self.engine.recommend(self._make_request())
        self.assertIsInstance(result, RecommendationResult)

    def test_recommend_result_size(self):
        result = self.engine.recommend(self._make_request(result_size=10, candidate_size=50))
        self.assertLessEqual(len(result.items), 10)

    def test_recommend_result_has_user_id(self):
        result = self.engine.recommend(self._make_request(user_id='user_42'))
        self.assertEqual(result.user_id, 'user_42')

    def test_recommend_result_has_request_id(self):
        result = self.engine.recommend(self._make_request())
        self.assertIsNotNone(result.request_id)

    # ── Parallel recall tests ──────────────────────────────────────────────────

    def test_parallel_recall_all_engines_called(self):
        """All registered recall engines are called when running in parallel."""
        from core.models import Item
        engine_a = _MockRecallEngine(items=[Item(item_id='a1', title='A1', category='cat', score=0.9)])
        engine_b = _MockRecallEngine(items=[Item(item_id='b1', title='B1', category='cat', score=0.8)])

        self.engine.register_recall_engine('engine_a', engine_a)
        self.engine.register_recall_engine('engine_b', engine_b)

        result = self.engine.recommend(self._make_request())
        self.assertIsInstance(result, RecommendationResult)
        self.assertEqual(engine_a.call_count, 1)
        self.assertEqual(engine_b.call_count, 1)

    def test_parallel_recall_merges_results(self):
        """Items from both recall engines appear in the result."""
        from core.models import Item
        engine_a = _MockRecallEngine(items=[Item(item_id='a1', title='A1', category='cat', score=0.9)])
        engine_b = _MockRecallEngine(items=[Item(item_id='b1', title='B1', category='cat', score=0.8)])

        eng = RecommendationEngine(recall_timeout_ms=2000)
        eng.register_recall_engine('engine_a', engine_a)
        eng.register_recall_engine('engine_b', engine_b)

        req = self._make_request(result_size=10)
        result = eng.recommend(req)
        item_ids = [item.item_id for item in result.items]
        self.assertIn('a1', item_ids)
        self.assertIn('b1', item_ids)

    def test_parallel_recall_timeout_skips_slow_engine(self):
        """A slow recall engine is skipped; fast engine results still returned."""
        from core.models import Item
        fast_engine = _MockRecallEngine(
            items=[Item(item_id='fast1', title='F1', category='cat', score=0.9)]
        )
        slow_engine = _MockRecallEngine(delay_s=2.0)  # 2 s >> 100 ms timeout

        eng = RecommendationEngine(recall_timeout_ms=100)
        eng.register_recall_engine('fast', fast_engine)
        eng.register_recall_engine('slow', slow_engine)

        result = eng.recommend(self._make_request(result_size=10))
        item_ids = [item.item_id for item in result.items]
        self.assertIn('fast1', item_ids)

    def test_parallel_recall_all_fail_uses_fallback(self):
        """When all recall engines fail, fallback items are returned."""
        bad_a = _MockRecallEngine(raise_exc=True)
        bad_b = _MockRecallEngine(raise_exc=True)

        eng = RecommendationEngine(recall_timeout_ms=500)
        eng.register_recall_engine('bad_a', bad_a)
        eng.register_recall_engine('bad_b', bad_b)

        result = eng.recommend(self._make_request(result_size=5))
        # Fallback should return items
        self.assertGreater(len(result.items), 0)

    def test_degradation_on_request_timeout(self):
        """When request_timeout_ms is very small, ranking degrades to score-based."""
        from unittest.mock import patch, MagicMock
        # Set an extremely short overall timeout so degradation is triggered
        eng = RecommendationEngine(request_timeout_ms=0.0)

        with patch('engine.ranking.score_ranking.ScoreRankingEngine') as mock_score_cls:
            mock_ranker = MagicMock()
            mock_ranker.rank.return_value = []
            mock_score_cls.return_value = mock_ranker

            result = eng.recommend(self._make_request())

        # ScoreRankingEngine should have been instantiated and used
        mock_score_cls.assert_called()
        self.assertIsInstance(result, RecommendationResult)


if __name__ == '__main__':
    unittest.main()

