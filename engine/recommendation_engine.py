import logging
import uuid
import time
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Main recommendation engine orchestrating the entire pipeline."""
    
    def __init__(self, config=None):
        """Initialize the recommendation engine."""
        self.config = config
        self.recall_engines = {}
        self.fusion_engine = None
        self.ranking_engine = None
        self.policy_manager = None
        self.cache = {}
        logger.info("Initialized RecommendationEngine")
    
    def recommend(self, request):
        """Generate recommendations through the complete pipeline."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"[{request_id}] Starting recommendation for user {request.user_context.user_id}")
            
            # Stage 1: Recall - retrieve candidate items
            logger.debug(f"[{request_id}] Stage 1: Recall")
            recalled_items = self._recall_stage(request, request_id)
            logger.debug(f"[{request_id}] Recall returned {len(recalled_items)} items")
            
            # Stage 2: Fusion - merge multiple recall paths
            logger.debug(f"[{request_id}] Stage 2: Fusion")
            fused_items = self._fusion_stage(recalled_items, request, request_id)
            logger.debug(f"[{request_id}] Fusion returned {len(fused_items)} items")
            
            # Stage 3: Ranking - rank items by relevance
            logger.debug(f"[{request_id}] Stage 3: Ranking")
            ranked_items = self._ranking_stage(fused_items, request, request_id)
            logger.debug(f"[{request_id}] Ranking returned {len(ranked_items)} items")
            
            # Stage 4: Business Rules - apply business constraints
            logger.debug(f"[{request_id}] Stage 4: Business Rules")
            final_items = self._business_rules_stage(ranked_items, request, request_id)
            logger.debug(f"[{request_id}] Business rules returned {len(final_items)} items")
            
            # Limit to result size
            result_size = getattr(request, 'result_size', 20)
            final_items = final_items[:result_size]
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create result
            from core.models import RecommendationResult
            result = RecommendationResult(
                request_id=request_id,
                items=final_items,
                user_id=request.user_context.user_id,
                processing_time_ms=processing_time_ms
            )
            
            logger.info(f"[{request_id}] Completed in {processing_time_ms:.2f}ms with {len(final_items)} results")
            return result
            
        except Exception as e:
            logger.error(f"[{request_id}] Error: {str(e)}", exc_info=True)
            from core.models import RecommendationResult
            return RecommendationResult(
                request_id=request_id,
                items=[],
                user_id=getattr(getattr(request, 'user_context', None), 'user_id', 'unknown')
            )
    
    def _recall_stage(self, request, request_id: str) -> List:
        """Retrieve candidate items from various sources."""
        items = []
        candidate_size = getattr(request, 'candidate_size', 100)

        if self.recall_engines:
            for name, engine in self.recall_engines.items():
                try:
                    recalled = engine.recall(request)
                    items.extend(recalled)
                    logger.debug(f"[{request_id}] Recall engine '{name}' returned {len(recalled)} items")
                except Exception as e:
                    logger.warning(f"[{request_id}] Recall engine '{name}' failed: {e}")

        if not items:
            from core.models import Item, ItemType, RecommendationSource
            for i in range(candidate_size):
                item = Item(
                    item_id=f"item_{i}",
                    title=f"Product {i}",
                    category="electronics",
                    item_type=ItemType.PRODUCT,
                    score=0.5 + (i % 50) / 100,
                    source=RecommendationSource.RECALL
                )
                items.append(item)

        logger.debug(f"[{request_id}] Recall stage: retrieved {len(items)} items")
        return items
    
    def _fusion_stage(self, items: List, request, request_id: str) -> List:
        """Merge multiple recall paths."""
        logger.debug(f"[{request_id}] Fusion stage: processing {len(items)} items")
        if self.fusion_engine:
            return self.fusion_engine.fuse(items)
        return items
    
    def _ranking_stage(self, items: List, request, request_id: str) -> List:
        """Rank items by relevance."""
        logger.debug(f"[{request_id}] Ranking stage: sorting {len(items)} items")
        
        if self.ranking_engine:
            return self.ranking_engine.rank(items, request.user_context)
        
        # Default sorting by score
        return sorted(items, key=lambda x: getattr(x, 'score', 0), reverse=True)
    
    def _business_rules_stage(self, items: List, request, request_id: str) -> List:
        """Apply business rules and filters."""
        logger.debug(f"[{request_id}] Business rules stage: filtering {len(items)} items")
        
        if self.policy_manager:
            context = {
                'user_context': request.user_context,
                'items': items
            }
            result_context = self.policy_manager.execute_stage_policies(
                'business_rules',
                context,
                items
            )
            return result_context if isinstance(result_context, list) else items
        
        return items
    
    def register_recall_engine(self, name: str, engine):
        """Register recall engine."""
        self.recall_engines[name] = engine
        logger.info(f"Registered recall engine: {name}")
    
    def register_fusion_engine(self, engine):
        """Register fusion engine."""
        self.fusion_engine = engine
        logger.info("Registered fusion engine")
    
    def register_ranking_engine(self, engine):
        """Register ranking engine."""
        self.ranking_engine = engine
        logger.info("Registered ranking engine")
    
    def register_policy_manager(self, manager):
        """Register policy manager."""
        self.policy_manager = manager
        logger.info("Registered policy manager")


class Recall:
    """Recall stage implementation."""
    
    def __init__(self, user_id):
        self.user_id = user_id
    
    def get_recommendations(self):
        """Retrieve recall items."""
        logger.debug(f"Recall: getting recommendations for user {self.user_id}")
        return []


class Fusion:
    """Fusion stage implementation."""
    
    def __init__(self, recall_items):
        self.recall_items = recall_items
    
    def fuse(self):
        """Merge various recall sources."""
        logger.debug(f"Fusion: merging {len(self.recall_items)} items")
        return self.recall_items


class Ranking:
    """Ranking stage implementation."""
    
    def __init__(self, fused_items):
        self.fused_items = fused_items
    
    def rank(self):
        """Rank items."""
        logger.debug(f"Ranking: ranking {len(self.fused_items)} items")
        return sorted(self.fused_items, key=lambda x: x.get('score', 0), reverse=True)


class BusinessRules:
    """Business rules stage implementation."""
    
    def __init__(self, ranked_items):
        self.ranked_items = ranked_items
    
    def apply_rules(self):
        """Apply business-specific logic."""
        logger.debug(f"BusinessRules: applying rules to {len(self.ranked_items)} items")
        return self.ranked_items


class RecommendationOrchestrator:
    """Orchestrator for the entire recommendation pipeline."""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.engine = None
    
    def orchestrate(self, request):
        """Execute the complete recommendation pipeline."""
        if not self.engine:
            self.engine = RecommendationEngine()
        
        return self.engine.recommend(request)