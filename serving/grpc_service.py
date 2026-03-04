"""
gRPC serving layer for EasyRec-Extended.

Provides a RecommendServiceServicer that exposes the same functionality as the
REST API over gRPC for low-latency production use.

Usage::

    from serving.grpc_service import serve_grpc
    serve_grpc(port=50051, server=recommendation_server, model_manager=manager)
"""
import logging
from concurrent import futures

import grpc

from serving import recommendation_pb2 as pb2
from serving import recommendation_pb2_grpc as pb2_grpc
from serving.health_check import health_check as _health_check

logger = logging.getLogger(__name__)


class RecommendServiceServicer(pb2_grpc.RecommendServiceServicer):
    """Implements the RecommendService gRPC interface."""

    def __init__(self, server=None, model_manager=None):
        """
        Args:
            server: A :class:`online.serving.RecommendationServer` instance.
            model_manager: A :class:`easyrec_extended.model_manager.ModelManager`
                instance (optional).
        """
        self._server = server
        self._model_manager = model_manager

    # ── Recommend ─────────────────────────────────────────────────────────────

    def Recommend(self, request, context):  # noqa: N802
        """Delegate to RecommendationServer and return a RecommendResponse."""
        if self._server is None:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("serving backend not initialised")
            return pb2.RecommendResponse(
                status="error", error="serving backend not initialised"
            )

        result = self._server.get_recommendations(
            user_id=request.user_id,
            result_size=request.result_size or None,
            candidate_size=request.candidate_size or None,
            filters=dict(request.filters) or None,
            policies=list(request.policies) or None,
            business_rules=list(request.business_rules) or None,
        )

        items = [
            pb2.RecommendedItem(
                item_id=str(item.get("item_id", "")),
                title=str(item.get("title", "")),
                category=str(item.get("category", "")),
                score=float(item.get("score", 0.0)),
                source=str(item.get("source", "")),
            )
            for item in result.get("items", [])
        ]

        return pb2.RecommendResponse(
            status=result.get("status", "error"),
            request_id=str(result.get("request_id", "")),
            user_id=str(result.get("user_id", request.user_id)),
            items=items,
            processing_time_ms=float(result.get("processing_time_ms", 0.0)),
            error=str(result.get("error", "")),
        )

    # ── HealthCheck ───────────────────────────────────────────────────────────

    def HealthCheck(self, request, context):  # noqa: N802
        """Return aggregate health of the service."""
        health = _health_check(model_manager=self._model_manager)
        return pb2.HealthCheckResponse(
            status=health.get("status", "degraded"),
            service=health.get("service", "EasyRec-Extended"),
            version=health.get("version", ""),
        )

    # ── ListModels ────────────────────────────────────────────────────────────

    def ListModels(self, request, context):  # noqa: N802
        """Return information about loaded model versions."""
        if self._model_manager is None:
            return pb2.ListModelsResponse(
                loaded_versions=[], active_version="", is_ready=False
            )
        return pb2.ListModelsResponse(
            loaded_versions=self._model_manager.loaded_versions,
            active_version=self._model_manager.active_version or "",
            is_ready=self._model_manager.is_ready(),
        )

    # ── ReloadModel ───────────────────────────────────────────────────────────

    def ReloadModel(self, request, context):  # noqa: N802
        """Hot-reload a model version."""
        if self._model_manager is None:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("model manager not initialised")
            return pb2.ReloadModelResponse(
                status="error", error="model manager not initialised"
            )

        if not request.version or not request.model_dir:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("version and model_dir are required")
            return pb2.ReloadModelResponse(
                status="error", error="version and model_dir are required"
            )

        try:
            self._model_manager.reload_version(request.version, request.model_dir)
            return pb2.ReloadModelResponse(
                status="success",
                message=f"Version '{request.version}' reloaded from {request.model_dir}",
            )
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return pb2.ReloadModelResponse(status="error", error=str(exc))
        except Exception as exc:  # pragma: no cover
            logger.error(f"Model reload failed: {exc}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return pb2.ReloadModelResponse(status="error", error=str(exc))


def serve_grpc(port: int = 50051, server=None, model_manager=None):
    """
    Start the gRPC server and block until it is stopped.

    Args:
        port: TCP port to listen on (default 50051).
        server: RecommendationServer instance shared with the HTTP layer.
        model_manager: ModelManager instance shared with the HTTP layer.
    """
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = RecommendServiceServicer(server=server, model_manager=model_manager)
    pb2_grpc.add_RecommendServiceServicer_to_server(servicer, grpc_server)
    listen_addr = f"[::]:{port}"
    grpc_server.add_insecure_port(listen_addr)
    grpc_server.start()
    logger.info(f"gRPC server started on port {port}")
    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("gRPC server shutting down…")
        grpc_server.stop(grace=5)
