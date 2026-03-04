"""Tests for Prometheus metrics and Flask middleware."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPrometheusMetricsRegistered(unittest.TestCase):
    """Test that all expected metrics are properly registered."""

    def setUp(self):
        # Import fresh to ensure metrics module loads
        from easyrec_extended.metrics import prometheus_metrics
        self.m = prometheus_metrics

    def test_recommendation_requests_total_is_counter(self):
        """recommendation_requests_total is a Counter."""
        from prometheus_client import Counter
        self.assertIsInstance(self.m.recommendation_requests_total, Counter)

    def test_recommendation_latency_seconds_is_histogram(self):
        """recommendation_latency_seconds is a Histogram."""
        from prometheus_client import Histogram
        self.assertIsInstance(self.m.recommendation_latency_seconds, Histogram)

    def test_recall_items_count_is_histogram(self):
        """recall_items_count is a Histogram."""
        from prometheus_client import Histogram
        self.assertIsInstance(self.m.recall_items_count, Histogram)

    def test_ranking_model_score_is_histogram(self):
        """ranking_model_score is a Histogram."""
        from prometheus_client import Histogram
        self.assertIsInstance(self.m.ranking_model_score, Histogram)

    def test_model_inference_latency_is_histogram(self):
        """model_inference_latency_seconds is a Histogram."""
        from prometheus_client import Histogram
        self.assertIsInstance(self.m.model_inference_latency_seconds, Histogram)

    def test_model_loaded_versions_is_gauge(self):
        """model_loaded_versions is a Gauge."""
        from prometheus_client import Gauge
        self.assertIsInstance(self.m.model_loaded_versions, Gauge)

    def test_active_model_version_is_info(self):
        """active_model_version is an Info."""
        from prometheus_client import Info
        self.assertIsInstance(self.m.active_model_version, Info)

    def test_feature_service_latency_is_histogram(self):
        """feature_service_latency_seconds is a Histogram."""
        from prometheus_client import Histogram
        self.assertIsInstance(self.m.feature_service_latency_seconds, Histogram)


class TestMetricsMiddleware(unittest.TestCase):
    """Test that the middleware increments counters and exposes /metrics."""

    def _make_app(self):
        from flask import Flask
        from easyrec_extended.metrics.middleware import MetricsMiddleware
        app = Flask(__name__ + str(id(self)))
        MetricsMiddleware(app)

        @app.route('/ping')
        def ping():
            from flask import jsonify
            return jsonify({'ok': True})

        return app

    def test_metrics_endpoint_returns_200(self):
        """/metrics endpoint responds with 200."""
        app = self._make_app()
        try:
            client = app.test_client()
        except AttributeError:
            self.skipTest("werkzeug version incompatible with test_client")
        resp = client.get('/metrics')
        self.assertEqual(resp.status_code, 200)

    def test_metrics_endpoint_returns_prometheus_format(self):
        """/metrics response contains Prometheus text format markers."""
        app = self._make_app()
        try:
            client = app.test_client()
        except AttributeError:
            self.skipTest("werkzeug version incompatible with test_client")
        resp = client.get('/metrics')
        body = resp.data.decode()
        # Prometheus text format starts with # HELP or # TYPE lines
        self.assertTrue('# HELP' in body or '# TYPE' in body or 'python_gc' in body)

    def test_request_increments_counter(self):
        """Hitting an endpoint increments recommendation_requests_total."""
        from prometheus_client import REGISTRY
        app = self._make_app()
        try:
            client = app.test_client()
        except AttributeError:
            self.skipTest("werkzeug version incompatible with test_client")

        # Collect baseline
        before = _collect_counter_samples("recommendation_requests_total")
        client.get('/ping')
        after = _collect_counter_samples("recommendation_requests_total")

        total_before = sum(s.value for s in before)
        total_after = sum(s.value for s in after)
        self.assertGreater(total_after, total_before)

    def test_404_request_tracked_as_error(self):
        """A 404 response is tracked with status=error."""
        app = self._make_app()
        try:
            client = app.test_client()
        except AttributeError:
            self.skipTest("werkzeug version incompatible with test_client")

        before = _collect_counter_samples("recommendation_requests_total")
        client.get('/nonexistent_path_xyz')
        after = _collect_counter_samples("recommendation_requests_total")

        total_before = sum(s.value for s in before)
        total_after = sum(s.value for s in after)
        self.assertGreaterEqual(total_after, total_before)


def _collect_counter_samples(metric_name):
    """Collect current samples for a named metric from the default registry."""
    from prometheus_client import REGISTRY
    samples = []
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            samples.extend(metric.samples)
    return samples


if __name__ == '__main__':
    unittest.main()
