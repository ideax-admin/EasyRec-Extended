"""
gRPC service implementation for EasyRec-Extended.

Exposes the RecommendationService defined in
``serving/protos/recommendation.proto`` over gRPC.

Usage::

    from serving.grpc_service import serve
    serve(port=50051)
"""
import logging
import time
import uuid
from concurrent import futures

import grpc

from serving.protos import recommendation_pb2
from serving.protos import recommendation_pb2_grpc

logger = logging.getLogger(__name__)

# Lazy module-level references set by register_grpc_components()
_server_instance = None
_model_manager = None
_config = None


def register_grpc_components(server=None, model_manager=None, config=None):
    """Bind serving components to the gRPC servicer.

    Args:
        server: :class:`online.serving.RecommendationServer` instance.
        model_manager: :class:`easyrec_extended.model_manager.ModelManager` instance.
        config: Application config object exposing ``SERVICE_NAME`` and ``VERSION``.
    """
    global _server_instance, _model_manager, _config
    _server_instance = server
    _model_manager = model_manager
    _config = config


class RecommendationServiceServicer(recommendation_pb2_grpc.RecommendationServiceServicer):
    """gRPC servicer that wraps the core recommendation stack."""

    def __init__(self, server=None, model_manager=None, config=None):
        """Create a servicer with optionally injected components.

        Args:
            server: :class:`online.serving.RecommendationServer` instance.
            model_manager: :class:`easyrec_extended.model_manager.ModelManager`.
            config: App config exposing ``SERVICE_NAME`` / ``VERSION``.
        """
        self._server = server or _server_instance
        self._model_manager = model_manager or _model_manager
        self._config = config or _config

    def GetRecommendations(self, request, context):
        """Handle a GetRecommendations RPC call."""
        request_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            "[%s] gRPC GetRecommendations for user '%s'",
            request_id,
            request.user_id,
        )

        if not request.user_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("user_id is required")
            return recommendation_pb2.RecommendResponse()

        if self._server is None:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Serving backend not initialised")
            return recommendation_pb2.RecommendResponse()

        try:
            result = self._server.get_recommendations(
                user_id=request.user_id,
                result_size=request.result_size or None,
                candidate_size=request.candidate_size or None,
                filters=dict(request.filters) if request.filters else None,
                policies=list(request.policies) if request.policies else None,
            )

            items = [
                recommendation_pb2.RecommendedItem(
                    item_id=item.get("item_id", ""),
                    title=item.get("title", ""),
                    category=item.get("category", ""),
                    score=float(item.get("score", 0.0)),
                    source=str(item.get("source", "")),
                )
                for item in result.get("items", [])
            ]

            processing_time_ms = (time.time() - start_time) * 1000
            logger.info(
                "[%s] gRPC GetRecommendations completed in %.2fms",
                request_id,
                processing_time_ms,
            )
            return recommendation_pb2.RecommendResponse(
                status=result.get("status", "success"),
                request_id=result.get("request_id", request_id),
                user_id=result.get("user_id", request.user_id),
                items=items,
                processing_time_ms=float(processing_time_ms),
            )
        except Exception as exc:
            logger.error("[%s] gRPC GetRecommendations error: %s", request_id, exc, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return recommendation_pb2.RecommendResponse()

    def HealthCheck(self, request, context):
        """Handle a HealthCheck RPC call."""
        service_name = "easyrec-extended"
        version = "1.0.0"

        if self._config is not None:
            service_name = getattr(self._config, "SERVICE_NAME", service_name)
            version = getattr(self._config, "VERSION", version)

        return recommendation_pb2.HealthCheckResponse(
            status="healthy",
            service=service_name,
            version=version,
        )

    def ReloadModel(self, request, context):
        """Handle a ReloadModel RPC call."""
        if not request.version or not request.model_dir:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("version and model_dir are required")
            return recommendation_pb2.ReloadModelResponse(
                status="error",
                message="version and model_dir are required",
            )

        if self._model_manager is None:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Model manager not initialised")
            return recommendation_pb2.ReloadModelResponse(
                status="error",
                message="Model manager not initialised",
            )

        try:
            self._model_manager.reload_version(request.version, request.model_dir)
            msg = f"Version '{request.version}' reloaded from {request.model_dir}"
            logger.info("gRPC ReloadModel: %s", msg)
            return recommendation_pb2.ReloadModelResponse(status="success", message=msg)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return recommendation_pb2.ReloadModelResponse(status="error", message=str(exc))
        except Exception as exc:
            logger.error("gRPC ReloadModel error: %s", exc, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return recommendation_pb2.ReloadModelResponse(status="error", message=str(exc))


def serve(port=50051, max_workers=10, server=None, model_manager=None, config=None):
    """Start the gRPC server and block until interrupted.

    Args:
        port: TCP port to listen on (default 50051).
        max_workers: Thread pool size for the gRPC server.
        server: Optional :class:`online.serving.RecommendationServer` to inject.
        model_manager: Optional :class:`easyrec_extended.model_manager.ModelManager`.
        config: Optional app config object.
    """
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = RecommendationServiceServicer(
        server=server,
        model_manager=model_manager,
        config=config,
    )
    recommendation_pb2_grpc.add_RecommendationServiceServicer_to_server(servicer, grpc_server)

    address = f"[::]:{port}"
    grpc_server.add_insecure_port(address)
    grpc_server.start()
    logger.info("gRPC server started on port %d", port)

    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("gRPC server shutting down...")
        grpc_server.stop(grace=5)
        logger.info("gRPC server stopped")
