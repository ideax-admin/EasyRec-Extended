import logging
import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from engine.parallel_executor import ParallelStageExecutor

logger = logging.getLogger(__name__)

# Default timeouts (milliseconds)
_DEFAULT_RECALL_TIMEOUT_MS = 500.0
_DEFAULT_REQUEST_TIMEOUT_MS = 2000.0


class RecommendationEngine:
    """Main recommendation engine orchestrating the entire pipeline."""

    def __init__(self, config=None, model_manager=None, feature_service=None,
                 recall_timeout_ms: float = _DEFAULT_RECALL_TIMEOUT_MS,
                 request_timeout_ms: float = _DEFAULT_REQUEST_TIMEOUT_MS):
        """Initialize the recommendation engine.

        Args:
            config: Optional configuration object.
            model_manager: Optional :class:`easyrec_extended.model_manager.ModelManager`
                instance.  When provided and ready, model-based ranking is used.
            feature_service: Optional :class:`easyrec_extended.features.feature_service.FeatureService`
                instance for feature enrichment.
            recall_timeout_ms: Per-recall-engine timeout in milliseconds
                (default 500 ms).
            request_timeout_ms: Overall pipeline timeout in milliseconds
                (default 2000 ms).  When exceeded, ranking degrades to
                :class:`engine.ranking.score_ranking.ScoreRankingEngine`.
        """
        self.config = config
        self.model_manager = model_manager
        self.feature_service = feature_service
        self.recall_engines = {}
        self.fusion_engine = None
        self.ranking_engine = None
        self.policy_manager = None
        self.cache = {}
        self.recall_timeout_ms = recall_timeout_ms
        self.request_timeout_ms = request_timeout_ms
        self._parallel_executor = ParallelStageExecutor()
        logger.info("Initialized RecommendationEngine")
    
    def recommend(self, request):
        """Generate recommendations through the complete pipeline."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"[{request_id}] Starting recommendation for user {request.user_context.user_id}")
            
            def _elapsed_ms():
                return (time.time() - start_time) * 1000

            # Stage 1: Recall - retrieve candidate items
            logger.debug(f"[{request_id}] Stage 1: Recall")
            recalled_items = self._recall_stage(request, request_id)
            logger.debug(f"[{request_id}] Recall returned {len(recalled_items)} items")
            
            # Stage 2: Feature Enrichment - enrich items with feature service
            logger.debug(f"[{request_id}] Stage 2: Feature Enrichment")
            enriched_items = self._feature_enrichment_stage(recalled_items, request, request_id)
            logger.debug(f"[{request_id}] Feature enrichment returned {len(enriched_items)} items")

            # Stage 3: Fusion - merge multiple recall paths
            logger.debug(f"[{request_id}] Stage 3: Fusion")
            fused_items = self._fusion_stage(enriched_items, request, request_id)
            logger.debug(f"[{request_id}] Fusion returned {len(fused_items)} items")
            
            # Stage 4: Ranking - rank items by relevance
            # Degrade to ScoreRankingEngine when overall timeout is approaching
            logger.debug(f"[{request_id}] Stage 4: Ranking")
            if _elapsed_ms() > self.request_timeout_ms:
                logger.warning(
                    f"[{request_id}] Request timeout ({self.request_timeout_ms}ms) reached "
                    "before ranking stage; degrading to ScoreRankingEngine"
                )
                from engine.ranking.score_ranking import ScoreRankingEngine
                ranked_items = ScoreRankingEngine().rank(fused_items)
            else:
                ranked_items = self._ranking_stage(fused_items, request, request_id)
            logger.debug(f"[{request_id}] Ranking returned {len(ranked_items)} items")
            
            # Stage 5: Business Rules - apply business constraints
            logger.debug(f"[{request_id}] Stage 5: Business Rules")
            final_items = self._business_rules_stage(ranked_items, request, request_id)
            logger.debug(f"[{request_id}] Business rules returned {len(final_items)} items")
            
            # Limit to result size
            result_size = getattr(request, 'result_size', 20)
            final_items = final_items[:result_size]
            
            processing_time_ms = _elapsed_ms()
            
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
        """Retrieve candidate items from various sources.

        When multiple recall engines are registered they are executed in
        **parallel** using :class:`engine.parallel_executor.ParallelStageExecutor`.
        Each engine has at most ``recall_timeout_ms`` milliseconds to return
        results; slow or failing engines are skipped.  If *all* engines fail or
        time out the pipeline falls back to
        :class:`engine.recall.fallback_recall.FallbackRecallEngine`.

        When a :class:`ModelManager` is available and ready the engine also
        runs an :class:`engine.recall.embedding_recall.EmbeddingRecallEngine`
        alongside the registered engines (unless one is already registered
        under the ``'embedding'`` key).
        """
        items = []
        callables = []

        # Prefer embedding-based recall when a model is available
        if (
            self.model_manager is not None
            and self.model_manager.is_ready()
            and 'embedding' not in self.recall_engines
        ):
            try:
                from engine.recall.embedding_recall import EmbeddingRecallEngine
                embedding_engine = EmbeddingRecallEngine(
                    model_manager=self.model_manager,
                    feature_service=self.feature_service,
                )
                callables.append(('embedding', embedding_engine.recall, request))
            except Exception as e:
                logger.warning(f"[{request_id}] Could not create EmbeddingRecallEngine: {e}")

        for name, engine in self.recall_engines.items():
            callables.append((name, engine.recall, request))

        if callables:
            recall_metrics: Dict[str, Any] = {}
            results = self._parallel_executor.execute_parallel(
                callables, timeout_ms=self.recall_timeout_ms
            )
            for name, result, elapsed_ms in results:
                recall_metrics[name] = {
                    'count': len(result) if result is not None else 0,
                    'elapsed_ms': elapsed_ms,
                    'success': result is not None,
                }
                if result is not None:
                    items.extend(result)
                    logger.debug(
                        f"[{request_id}] Recall engine '{name}' returned "
                        f"{len(result)} items in {elapsed_ms:.1f}ms"
                    )
            logger.debug(f"[{request_id}] Recall metrics: {recall_metrics}")

        if not items:
            from engine.recall.fallback_recall import FallbackRecallEngine
            fallback = FallbackRecallEngine()
            items = fallback.recall(request)
            logger.debug(f"[{request_id}] Fallback recall returned {len(items)} items")

        logger.debug(f"[{request_id}] Recall stage: retrieved {len(items)} items")
        return items
    
    def _fusion_stage(self, items: List, request, request_id: str) -> List:
        """Merge multiple recall paths."""
        logger.debug(f"[{request_id}] Fusion stage: processing {len(items)} items")
        if self.fusion_engine:
            return self.fusion_engine.fuse(items)
        return items

    def _feature_enrichment_stage(self, items: List, request, request_id: str) -> List:
        """Enrich candidate items with features from the feature service.

        When a :class:`FeatureService` is configured, item features are fetched
        and stored in ``item.features`` for downstream ranking.

        Args:
            items: Candidate items to enrich.
            request: The recommendation request.
            request_id: Request identifier for logging.

        Returns:
            Items with ``features`` populated (unchanged when no service).
        """
        if self.feature_service is None:
            return items

        item_ids = [item.item_id for item in items]
        try:
            item_feats_map = self.feature_service.get_item_features(item_ids)
            for item in items:
                feats = item_feats_map.get(item.item_id, {})
                if feats:
                    item.features = {**getattr(item, 'features', {}), **feats}
        except Exception as e:
            logger.warning(f"[{request_id}] Feature enrichment failed: {e}")

        logger.debug(f"[{request_id}] Feature enrichment stage: processed {len(items)} items")
        return items

    def _ranking_stage(self, items: List, request, request_id: str) -> List:
        """Rank items by relevance.

        Uses the registered :attr:`ranking_engine` if present.  If a
        :class:`ModelManager` is available and ready, a
        :class:`engine.ranking.model_ranking.ModelRankingEngine` is used
        automatically.  Falls back to score-attribute sorting otherwise.
        """
        logger.debug(f"[{request_id}] Ranking stage: sorting {len(items)} items")

        if self.ranking_engine:
            return self.ranking_engine.rank(items, request.user_context)

        if self.model_manager is not None and self.model_manager.is_ready():
            try:
                from engine.ranking.model_ranking import ModelRankingEngine
                model_ranker = ModelRankingEngine(
                    model_manager=self.model_manager,
                    feature_service=self.feature_service,
                )
                return model_ranker.rank(items, request.user_context)
            except Exception as e:
                logger.warning(f"[{request_id}] ModelRankingEngine failed: {e}; falling back")

        # Default: sort by existing score attribute
        from engine.ranking.score_ranking import ScoreRankingEngine
        return ScoreRankingEngine().rank(items)
    
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