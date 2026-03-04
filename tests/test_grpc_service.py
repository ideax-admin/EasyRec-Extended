"""Tests for the gRPC servicer methods (no running server required)."""
import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.protos import recommendation_pb2
from serving.grpc_service import RecommendationServiceServicer


def _make_servicer(server=None, model_manager=None, config=None):
    """Helper to create a servicer with mock components."""
    return RecommendationServiceServicer(
        server=server,
        model_manager=model_manager,
        config=config,
    )


def _make_context():
    """Return a MagicMock grpc context."""
    ctx = MagicMock()
    return ctx


class TestGetRecommendations(unittest.TestCase):
    """Tests for RecommendationServiceServicer.GetRecommendations."""

    def _make_server(self, items=None):
        server = MagicMock()
        server.get_recommendations.return_value = {
            'status': 'success',
            'request_id': 'req-001',
            'user_id': 'user_42',
            'items': items or [
                {'item_id': 'item_1', 'title': 'Item 1', 'category': 'cat', 'score': 0.9,
                 'source': 'recall'},
            ],
            'processing_time_ms': 12.5,
        }
        return server

    def test_get_recommendations_returns_response(self):
        """GetRecommendations returns a response with items."""
        servicer = _make_servicer(server=self._make_server())
        req = recommendation_pb2.RecommendRequest(user_id='user_42', result_size=5)
        ctx = _make_context()
        resp = servicer.GetRecommendations(req, ctx)
        self.assertEqual(resp.status, 'success')
        self.assertEqual(resp.user_id, 'user_42')
        self.assertEqual(len(resp.items), 1)
        self.assertEqual(resp.items[0].item_id, 'item_1')

    def test_get_recommendations_missing_user_id_sets_error(self):
        """GetRecommendations with empty user_id sets INVALID_ARGUMENT."""
        import grpc
        servicer = _make_servicer(server=self._make_server())
        req = recommendation_pb2.RecommendRequest(user_id='')
        ctx = _make_context()
        resp = servicer.GetRecommendations(req, ctx)
        ctx.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(resp.status, '')

    def test_get_recommendations_no_server_sets_unavailable(self):
        """GetRecommendations with no server set returns UNAVAILABLE."""
        import grpc
        servicer = _make_servicer()
        req = recommendation_pb2.RecommendRequest(user_id='user_1')
        ctx = _make_context()
        servicer.GetRecommendations(req, ctx)
        ctx.set_code.assert_called_once_with(grpc.StatusCode.UNAVAILABLE)

    def test_get_recommendations_response_has_request_id(self):
        """GetRecommendations response includes a request_id."""
        servicer = _make_servicer(server=self._make_server())
        req = recommendation_pb2.RecommendRequest(user_id='user_42')
        ctx = _make_context()
        resp = servicer.GetRecommendations(req, ctx)
        self.assertNotEqual(resp.request_id, '')

    def test_get_recommendations_processing_time_positive(self):
        """GetRecommendations response processing_time_ms is positive."""
        servicer = _make_servicer(server=self._make_server())
        req = recommendation_pb2.RecommendRequest(user_id='user_42')
        ctx = _make_context()
        resp = servicer.GetRecommendations(req, ctx)
        self.assertGreaterEqual(resp.processing_time_ms, 0.0)


class TestHealthCheck(unittest.TestCase):
    """Tests for RecommendationServiceServicer.HealthCheck."""

    def test_health_check_returns_healthy(self):
        """HealthCheck returns healthy status."""
        servicer = _make_servicer()
        req = recommendation_pb2.HealthCheckRequest()
        ctx = _make_context()
        resp = servicer.HealthCheck(req, ctx)
        self.assertEqual(resp.status, 'healthy')

    def test_health_check_returns_service_name(self):
        """HealthCheck returns service name from config."""
        config = MagicMock()
        config.SERVICE_NAME = 'my-service'
        config.VERSION = '2.0.0'
        servicer = _make_servicer(config=config)
        req = recommendation_pb2.HealthCheckRequest()
        ctx = _make_context()
        resp = servicer.HealthCheck(req, ctx)
        self.assertEqual(resp.service, 'my-service')
        self.assertEqual(resp.version, '2.0.0')

    def test_health_check_defaults_when_no_config(self):
        """HealthCheck uses sensible defaults when no config is provided."""
        servicer = _make_servicer()
        req = recommendation_pb2.HealthCheckRequest()
        ctx = _make_context()
        resp = servicer.HealthCheck(req, ctx)
        self.assertIsInstance(resp.service, str)
        self.assertIsInstance(resp.version, str)


class TestReloadModel(unittest.TestCase):
    """Tests for RecommendationServiceServicer.ReloadModel."""

    def test_reload_model_missing_version_returns_error(self):
        """ReloadModel with missing version field returns error response."""
        import grpc
        servicer = _make_servicer(model_manager=MagicMock())
        req = recommendation_pb2.ReloadModelRequest(version='', model_dir='/tmp/model')
        ctx = _make_context()
        resp = servicer.ReloadModel(req, ctx)
        self.assertEqual(resp.status, 'error')
        ctx.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)

    def test_reload_model_missing_model_dir_returns_error(self):
        """ReloadModel with missing model_dir returns error response."""
        import grpc
        servicer = _make_servicer(model_manager=MagicMock())
        req = recommendation_pb2.ReloadModelRequest(version='v1', model_dir='')
        ctx = _make_context()
        resp = servicer.ReloadModel(req, ctx)
        self.assertEqual(resp.status, 'error')
        ctx.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)

    def test_reload_model_no_manager_returns_unavailable(self):
        """ReloadModel with no model_manager set returns UNAVAILABLE."""
        import grpc
        servicer = _make_servicer()
        req = recommendation_pb2.ReloadModelRequest(version='v1', model_dir='/tmp/model')
        ctx = _make_context()
        resp = servicer.ReloadModel(req, ctx)
        ctx.set_code.assert_called_once_with(grpc.StatusCode.UNAVAILABLE)

    def test_reload_model_success(self):
        """ReloadModel succeeds when model_manager.reload_version does not raise."""
        mm = MagicMock()
        servicer = _make_servicer(model_manager=mm)
        req = recommendation_pb2.ReloadModelRequest(version='v1', model_dir='/tmp/model')
        ctx = _make_context()
        resp = servicer.ReloadModel(req, ctx)
        self.assertEqual(resp.status, 'success')
        mm.reload_version.assert_called_once_with('v1', '/tmp/model')

    def test_reload_model_value_error_returns_not_found(self):
        """ReloadModel with ValueError from model_manager returns NOT_FOUND."""
        import grpc
        mm = MagicMock()
        mm.reload_version.side_effect = ValueError("not found")
        servicer = _make_servicer(model_manager=mm)
        req = recommendation_pb2.ReloadModelRequest(version='v9', model_dir='/tmp/model')
        ctx = _make_context()
        resp = servicer.ReloadModel(req, ctx)
        self.assertEqual(resp.status, 'error')
        ctx.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)


if __name__ == '__main__':
    unittest.main()
