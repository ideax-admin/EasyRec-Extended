import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.recommendation_engine import RecommendationEngine
from core.models import RecommendationRequest, UserContext, RecommendationResult
from datetime import datetime
import uuid


class TestRecommendationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RecommendationEngine()

    def test_recommend_returns_result(self):
        user_context = UserContext(user_id='user_1', session_id=str(uuid.uuid4()), timestamp=datetime.now())
        request = RecommendationRequest(user_context=user_context, candidate_size=10, result_size=5)
        result = self.engine.recommend(request)
        self.assertIsInstance(result, RecommendationResult)

    def test_recommend_result_size(self):
        user_context = UserContext(user_id='user_1', session_id=str(uuid.uuid4()), timestamp=datetime.now())
        request = RecommendationRequest(user_context=user_context, candidate_size=50, result_size=10)
        result = self.engine.recommend(request)
        self.assertLessEqual(len(result.items), 10)

    def test_recommend_result_has_user_id(self):
        user_context = UserContext(user_id='user_42', session_id=str(uuid.uuid4()), timestamp=datetime.now())
        request = RecommendationRequest(user_context=user_context, candidate_size=10, result_size=5)
        result = self.engine.recommend(request)
        self.assertEqual(result.user_id, 'user_42')

    def test_recommend_result_has_request_id(self):
        user_context = UserContext(user_id='user_1', session_id=str(uuid.uuid4()), timestamp=datetime.now())
        request = RecommendationRequest(user_context=user_context, candidate_size=10, result_size=5)
        result = self.engine.recommend(request)
        self.assertIsNotNone(result.request_id)


if __name__ == '__main__':
    unittest.main()
