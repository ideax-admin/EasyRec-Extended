import logging
import os
import uuid
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def _get_metrics():
    """Return a dict of Prometheus metric objects, or None if unavailable."""
    try:
        from easyrec_extended.metrics.prometheus_metrics import (
            recommendation_requests_total,
            recommendation_latency_seconds,
            recall_items_count,
        )
        return {
            "requests_total": recommendation_requests_total,
            "total_latency": recommendation_latency_seconds.labels(stage="total"),
            "recall_latency": recommendation_latency_seconds.labels(stage="recall"),
            "recall_count": recall_items_count,
            "fusion_latency": recommendation_latency_seconds.labels(stage="fusion"),
            "ranking_latency": recommendation_latency_seconds.labels(stage="ranking"),
            "business_rules_latency": recommendation_latency_seconds.labels(
                stage="business_rules"
            ),
        }
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Pipeline tracing dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StageTrace:
    """Timing and metadata for a single pipeline stage.

    Attributes:
        name: Stage name (e.g. ``"recall"``, ``"ranking"``).
        start_time: Unix timestamp when the stage started.
        end_time: Unix timestamp when the stage ended.
        item_count: Number of items at stage output.
        metadata: Arbitrary key-value metadata for the stage.
    """

    name: str
    start_time: float
    end_time: float
    item_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Stage duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000


@dataclass
class PipelineTrace:
    """End-to-end trace of a recommendation request.

    Attributes:
        request_id: Unique identifier for the request.
        stages: Ordered list of :class:`StageTrace` instances.
        total_duration_ms: Total pipeline duration in milliseconds.
        model_version_used: Model version used for ranking (if any).
        experiment_assignment: A/B experiment assignment, e.g. ``"control"``.
    """

    request_id: str
    stages: List[StageTrace] = field(default_factory=list)
    total_duration_ms: float = 0.0
    model_version_used: Optional[str] = None
    experiment_assignment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the trace to a plain dict suitable for JSON responses."""
        return {
            "request_id": self.request_id,
            "total_duration_ms": round(self.total_duration_ms, 3),
            "model_version_used": self.model_version_used,
            "experiment_assignment": self.experiment_assignment,
            "stages": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 3),
                    "item_count": s.item_count,
                    "metadata": s.metadata,
                }
                for s in self.stages
            ],
        }


class RecommendationEngine:
    """Main recommendation engine orchestrating the entire pipeline."""

    def __init__(self, config=None, model_manager=None, feature_service=None,
                 experiment_manager=None):
        """Initialize the recommendation engine.

        Args:
            config: Optional configuration object.
            model_manager: Optional :class:`easyrec_extended.model_manager.ModelManager`
                instance.  When provided and ready, model-based ranking is used.
            feature_service: Optional :class:`easyrec_extended.features.feature_service.FeatureService`
                instance for feature enrichment.
            experiment_manager: Optional
                :class:`easyrec_extended.experiment.experiment_manager.ExperimentManager`
                instance for A/B experiment support.
        """
        self.config = config
        self.model_manager = model_manager
        self.feature_service = feature_service
        self.experiment_manager = experiment_manager
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
        enable_trace = os.environ.get("ENABLE_TRACE", "false").lower() == "true"
        trace = PipelineTrace(request_id=request_id) if enable_trace else None

        # Resolve experiment assignment (if any)
        experiment_assignment = None
        experiment_version = None
        experiment_name = getattr(request, "experiment_name", None)
        if experiment_name and self.experiment_manager is not None:
            try:
                user_id = request.user_context.user_id
                experiment_assignment = self.experiment_manager.assign_user(
                    user_id, experiment_name
                )
                experiment_version = self.experiment_manager.get_model_version_for_user(
                    user_id, experiment_name
                )
                if trace:
                    trace.experiment_assignment = experiment_assignment
                logger.info(
                    "[%s] Experiment '%s': user '%s' assigned to '%s' (version=%s)",
                    request_id, experiment_name, user_id,
                    experiment_assignment, experiment_version,
                )
            except Exception as exc:
                logger.warning("[%s] Experiment assignment failed: %s", request_id, exc)

        # Emit metrics lazily to avoid hard dependency on prometheus_client
        _metrics = _get_metrics()

        try:
            logger.info(f"[{request_id}] Starting recommendation for user {request.user_context.user_id}")

            # Stage 1: Recall - retrieve candidate items
            logger.debug(f"[{request_id}] Stage 1: Recall")
            _t0 = time.time()
            recalled_items = self._recall_stage(request, request_id)
            _t1 = time.time()
            logger.debug(f"[{request_id}] Recall returned {len(recalled_items)} items")
            if _metrics:
                _metrics["recall_latency"].observe(_t1 - _t0)
                _metrics["recall_count"].observe(len(recalled_items))
            if trace:
                trace.stages.append(StageTrace("recall", _t0, _t1, len(recalled_items)))

            # Stage 2: Feature Enrichment - enrich items with feature service
            logger.debug(f"[{request_id}] Stage 2: Feature Enrichment")
            _t0 = time.time()
            enriched_items = self._feature_enrichment_stage(recalled_items, request, request_id)
            _t1 = time.time()
            logger.debug(f"[{request_id}] Feature enrichment returned {len(enriched_items)} items")
            if trace:
                trace.stages.append(StageTrace("feature_enrichment", _t0, _t1, len(enriched_items)))

            # Stage 3: Fusion - merge multiple recall paths
            logger.debug(f"[{request_id}] Stage 3: Fusion")
            _t0 = time.time()
            fused_items = self._fusion_stage(enriched_items, request, request_id)
            _t1 = time.time()
            logger.debug(f"[{request_id}] Fusion returned {len(fused_items)} items")
            if _metrics:
                _metrics["fusion_latency"].observe(_t1 - _t0)
            if trace:
                trace.stages.append(StageTrace("fusion", _t0, _t1, len(fused_items)))

            # Stage 4: Ranking - rank items by relevance
            logger.debug(f"[{request_id}] Stage 4: Ranking")
            _t0 = time.time()
            ranked_items = self._ranking_stage(
                fused_items, request, request_id,
                model_version=experiment_version,
            )
            _t1 = time.time()
            logger.debug(f"[{request_id}] Ranking returned {len(ranked_items)} items")
            if _metrics:
                _metrics["ranking_latency"].observe(_t1 - _t0)
            if trace:
                trace.stages.append(StageTrace("ranking", _t0, _t1, len(ranked_items)))

            # Stage 5: Business Rules - apply business constraints
            logger.debug(f"[{request_id}] Stage 5: Business Rules")
            _t0 = time.time()
            final_items = self._business_rules_stage(ranked_items, request, request_id)
            _t1 = time.time()
            logger.debug(f"[{request_id}] Business rules returned {len(final_items)} items")
            if _metrics:
                _metrics["business_rules_latency"].observe(_t1 - _t0)
            if trace:
                trace.stages.append(StageTrace("business_rules", _t0, _t1, len(final_items)))

            # Limit to result size
            result_size = getattr(request, 'result_size', 20)
            final_items = final_items[:result_size]

            processing_time_ms = (time.time() - start_time) * 1000
            if _metrics:
                _metrics["total_latency"].observe(processing_time_ms / 1000)
                _metrics["requests_total"].labels(status="success", endpoint="engine").inc()
            if trace:
                trace.total_duration_ms = processing_time_ms

            # Create result
            from core.models import RecommendationResult
            metadata = {}
            if trace:
                metadata["trace"] = trace.to_dict()
            result = RecommendationResult(
                request_id=request_id,
                items=final_items,
                user_id=request.user_context.user_id,
                processing_time_ms=processing_time_ms,
                metadata=metadata,
            )

            logger.info(f"[{request_id}] Completed in {processing_time_ms:.2f}ms with {len(final_items)} results")
            return result

        except Exception as e:
            logger.error(f"[{request_id}] Error: {str(e)}", exc_info=True)
            if _metrics:
                _metrics["requests_total"].labels(status="error", endpoint="engine").inc()
            from core.models import RecommendationResult
            return RecommendationResult(
                request_id=request_id,
                items=[],
                user_id=getattr(getattr(request, 'user_context', None), 'user_id', 'unknown')
            )
    
    def _recall_stage(self, request, request_id: str) -> List:
        """Retrieve candidate items from various sources.

        When a :class:`ModelManager` is available and ready the engine checks
        for a registered ``'embedding'`` recall engine first.  If no embedding
        recall engine is found it falls back to registered recall engines or,
        finally, to synthetic fallback items.
        """
        items = []
        candidate_size = getattr(request, 'candidate_size', 100)

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
                recalled = embedding_engine.recall(request)
                items.extend(recalled)
                logger.debug(
                    f"[{request_id}] EmbeddingRecallEngine returned {len(recalled)} items"
                )
            except Exception as e:
                logger.warning(f"[{request_id}] EmbeddingRecallEngine failed: {e}")

        if self.recall_engines:
            for name, engine in self.recall_engines.items():
                try:
                    recalled = engine.recall(request)
                    items.extend(recalled)
                    logger.debug(
                        f"[{request_id}] Recall engine '{name}' returned {len(recalled)} items"
                    )
                except Exception as e:
                    logger.warning(f"[{request_id}] Recall engine '{name}' failed: {e}")

        if not items:
            from engine.recall.fallback_recall import FallbackRecallEngine
            fallback = FallbackRecallEngine()
            items = fallback.recall(request)

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

    def _ranking_stage(self, items: List, request, request_id: str,
                       model_version: Optional[str] = None) -> List:
        """Rank items by relevance.

        Uses the registered :attr:`ranking_engine` if present.  If a
        :class:`ModelManager` is available and ready, a
        :class:`engine.ranking.model_ranking.ModelRankingEngine` is used
        automatically.  Falls back to score-attribute sorting otherwise.

        Args:
            items: Candidate items to rank.
            request: The recommendation request.
            request_id: Request identifier for logging.
            model_version: Optional model version override (e.g. from A/B experiment).
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
                    model_version=model_version,
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

    def register_experiment_manager(self, manager):
        """Register an :class:`ExperimentManager` for A/B experiment support.

        Args:
            manager: :class:`easyrec_extended.experiment.experiment_manager.ExperimentManager`
                instance.
        """
        self.experiment_manager = manager
        logger.info("Registered experiment manager")


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