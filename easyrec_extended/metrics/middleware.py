"""
Flask middleware for automatic Prometheus metrics collection.

Register with your Flask app::

    from easyrec_extended.metrics.middleware import MetricsMiddleware
    MetricsMiddleware(app)

This will:
- Count all requests (recommendation_requests_total).
- Record request latency (recommendation_latency_seconds with stage="total").
- Expose a ``/metrics`` endpoint returning Prometheus text format.
"""
import time
import logging

from flask import request, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from easyrec_extended.metrics.prometheus_metrics import (
    recommendation_requests_total,
    recommendation_latency_seconds,
)

logger = logging.getLogger(__name__)


class MetricsMiddleware:
    """Flask extension that instruments all HTTP endpoints with Prometheus metrics.

    Args:
        app: The :class:`flask.Flask` application instance to instrument.
    """

    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Register before/after request hooks and the /metrics endpoint.

        Args:
            app: :class:`flask.Flask` application instance.
        """
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        app.add_url_rule("/metrics", "prometheus_metrics", self._metrics_view)
        logger.info("Prometheus MetricsMiddleware registered")

    @staticmethod
    def _before_request():
        """Record request start time in the Flask ``g`` object."""
        from flask import g
        g.start_time = time.time()

    @staticmethod
    def _after_request(response):
        """Increment counters and record latency after each request.

        Args:
            response: The :class:`flask.Response` being returned.

        Returns:
            The same response object (unmodified).
        """
        from flask import g
        start_time = getattr(g, "start_time", None)
        if start_time is not None:
            elapsed = time.time() - start_time
            endpoint = request.path
            status = "success" if response.status_code < 400 else "error"
            recommendation_requests_total.labels(
                status=status, endpoint=endpoint
            ).inc()
            recommendation_latency_seconds.labels(stage="total").observe(elapsed)
        return response

    @staticmethod
    def _metrics_view():
        """Expose Prometheus metrics in text format.

        Returns:
            HTTP 200 response with ``text/plain; version=0.0.4`` content type.
        """
        data = generate_latest()
        return Response(data, status=200, mimetype=CONTENT_TYPE_LATEST)
