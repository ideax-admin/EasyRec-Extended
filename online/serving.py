import logging
from typing import Dict, Any
from datetime import datetime
import uuid
from core.models import RecommendationRequest, UserContext
from engine.recommendation_engine import RecommendationEngine
from core.config import get_config

logger = logging.getLogger(__name__)

class RecommendationServer:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.engine = RecommendationEngine(self.config)
        self.request_count = 0
        self.error_count = 0

    def get_recommendations(self, user_id: str, candidate_size: int = None, result_size: int = None,
                          filters: Dict[str, Any] = None, policies: list = None,
                          business_rules: list = None) -> Dict[str, Any]:
        self.request_count += 1
        try:
            candidate_size = candidate_size or self.config.DEFAULT_RECALL_SIZE
            result_size = result_size or self.config.DEFAULT_RESULT_SIZE
            user_context = UserContext(user_id=user_id, session_id=str(uuid.uuid4()), timestamp=datetime.now())
            request = RecommendationRequest(
                user_context=user_context,
                candidate_size=candidate_size,
                result_size=result_size,
                filters=filters or {},
                policies=policies or [],
                business_rules=business_rules or []
            )
            result = self.engine.recommend(request)
            logger.info(f'Successfully returned {len(result.items)} recommendations')
            return {
                'status': 'success',
                'request_id': result.request_id,
                'user_id': result.user_id,
                'items': [{'item_id': item.item_id, 'title': item.title, 'score': item.score} for item in result.items],
                'processing_time_ms': result.processing_time_ms
            }
        except Exception as e:
            self.error_count += 1
            logger.error(f'Error: {str(e)}')
            return {'status': 'error', 'error': str(e)}