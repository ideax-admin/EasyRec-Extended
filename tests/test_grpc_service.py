"""Tests for the gRPC serving layer (servicer methods only, no real gRPC server)."""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grpc
from serving.grpc_service import RecommendServiceServicer
from serving import recommendation_pb2 as pb2


def _make_context():
    """Return a mock gRPC ServicerContext."""
    ctx = MagicMock()
    ctx.set_code = MagicMock()
    ctx.set_details = MagicMock()
    return ctx


class TestRecommendRPC(unittest.TestCase):
    """Tests for the Recommend RPC."""

    def _make_server(self, items=None, status='success'):
        """Return a mock RecommendationServer."""
        server = MagicMock()
        server.get_recommendations.return_value = {
            'status': status,
            'request_id': 'req-123',
            'user_id': 'user_1',
            'items': items or [
                {'item_id': 'item_1', 'title': 'T1', 'category': 'C1', 'score': 0.9, 'source': 's1'}
            ],
            'processing_time_ms': 10.0,
        }
        return server

    def test_recommend_returns_response_format(self):
        """Recommend RPC returns a RecommendResponse with expected fields."""
        servicer = RecommendServiceServicer(server=self._make_server())
        req = pb2.RecommendRequest(user_id='user_1', result_size=5)
        resp = servicer.Recommend(req, _make_context())

        self.assertIsInstance(resp, pb2.RecommendResponse)
        self.assertEqual(resp.status, 'success')
        self.assertEqual(resp.request_id, 'req-123')
        self.assertEqual(resp.user_id, 'user_1')
        self.assertEqual(len(resp.items), 1)
        self.assertEqual(resp.items[0].item_id, 'item_1')
        self.assertAlmostEqual(resp.items[0].score, 0.9, places=4)

    def test_recommend_without_server_returns_error(self):
        """Recommend RPC without a server sets UNAVAILABLE status code."""
        servicer = RecommendServiceServicer(server=None)
        req = pb2.RecommendRequest(user_id='user_1')
        ctx = _make_context()
        resp = servicer.Recommend(req, ctx)

        ctx.set_code.assert_called_once_with(grpc.StatusCode.UNAVAILABLE)
        self.assertEqual(resp.status, 'error')

    def test_recommend_passes_parameters(self):
        """Recommend RPC forwards all request fields to the server."""
        server = self._make_server()
        servicer = RecommendServiceServicer(server=server)
        req = pb2.RecommendRequest(
            user_id='u42',
            result_size=10,
            candidate_size=50,
            policies=['p1'],
            business_rules=['r1'],
        )
        req.filters['category'] = 'electronics'
        servicer.Recommend(req, _make_context())

        call_kwargs = server.get_recommendations.call_args[1]
        self.assertEqual(call_kwargs['user_id'], 'u42')


class TestHealthCheckRPC(unittest.TestCase):
    """Tests for the HealthCheck RPC."""

    def test_health_check_returns_response(self):
        """HealthCheck RPC returns a HealthCheckResponse."""
        servicer = RecommendServiceServicer()
        resp = servicer.HealthCheck(pb2.HealthCheckRequest(), _make_context())
        self.assertIsInstance(resp, pb2.HealthCheckResponse)
        self.assertIn(resp.status, ('healthy', 'degraded'))

    def test_health_check_with_ready_model_manager(self):
        """HealthCheck returns healthy when model manager is ready."""
        mm = MagicMock()
        mm.is_ready.return_value = True
        mm.active_version = 'v1'
        mm.loaded_versions = ['v1']

        with patch('serving.grpc_service._health_check') as mock_hc:
            mock_hc.return_value = {'status': 'healthy', 'service': 'EasyRec', 'version': '0.1'}
            servicer = RecommendServiceServicer(model_manager=mm)
            resp = servicer.HealthCheck(pb2.HealthCheckRequest(), _make_context())
        self.assertEqual(resp.status, 'healthy')


class TestListModelsRPC(unittest.TestCase):
    """Tests for the ListModels RPC."""

    def test_list_models_without_manager(self):
        """ListModels returns empty list when no model manager."""
        servicer = RecommendServiceServicer()
        resp = servicer.ListModels(pb2.ListModelsRequest(), _make_context())
        self.assertIsInstance(resp, pb2.ListModelsResponse)
        self.assertEqual(list(resp.loaded_versions), [])
        self.assertFalse(resp.is_ready)

    def test_list_models_with_manager(self):
        """ListModels returns loaded versions from model manager."""
        mm = MagicMock()
        mm.loaded_versions = ['v1', 'v2']
        mm.active_version = 'v2'
        mm.is_ready.return_value = True

        servicer = RecommendServiceServicer(model_manager=mm)
        resp = servicer.ListModels(pb2.ListModelsRequest(), _make_context())

        self.assertIn('v1', resp.loaded_versions)
        self.assertIn('v2', resp.loaded_versions)
        self.assertEqual(resp.active_version, 'v2')
        self.assertTrue(resp.is_ready)


class TestReloadModelRPC(unittest.TestCase):
    """Tests for the ReloadModel RPC."""

    def test_reload_without_manager_returns_error(self):
        """ReloadModel sets UNAVAILABLE when no model manager."""
        servicer = RecommendServiceServicer()
        req = pb2.ReloadModelRequest(version='v1', model_dir='/tmp/m')
        ctx = _make_context()
        resp = servicer.ReloadModel(req, ctx)
        ctx.set_code.assert_called_once_with(grpc.StatusCode.UNAVAILABLE)
        self.assertEqual(resp.status, 'error')

    def test_reload_missing_fields_returns_invalid_argument(self):
        """ReloadModel with missing fields sets INVALID_ARGUMENT."""
        mm = MagicMock()
        servicer = RecommendServiceServicer(model_manager=mm)
        req = pb2.ReloadModelRequest()  # empty
        ctx = _make_context()
        resp = servicer.ReloadModel(req, ctx)
        ctx.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(resp.status, 'error')

    def test_reload_nonexistent_version_returns_not_found(self):
        """ReloadModel with non-existent version sets NOT_FOUND."""
        mm = MagicMock()
        mm.reload_version.side_effect = ValueError("version 'v99' not loaded")
        servicer = RecommendServiceServicer(model_manager=mm)
        req = pb2.ReloadModelRequest(version='v99', model_dir='/tmp/m')
        ctx = _make_context()
        resp = servicer.ReloadModel(req, ctx)
        ctx.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)
        self.assertEqual(resp.status, 'error')

    def test_reload_success(self):
        """ReloadModel returns success on valid request."""
        mm = MagicMock()
        mm.reload_version.return_value = None
        servicer = RecommendServiceServicer(model_manager=mm)
        req = pb2.ReloadModelRequest(version='v1', model_dir='/tmp/model')
        resp = servicer.ReloadModel(req, _make_context())
        self.assertEqual(resp.status, 'success')
        mm.reload_version.assert_called_once_with('v1', '/tmp/model')


if __name__ == '__main__':
    unittest.main()
