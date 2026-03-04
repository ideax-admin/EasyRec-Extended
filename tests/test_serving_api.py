"""Tests for the Flask serving API endpoints."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_app():
    """Create a Flask test app with the api_bp blueprint registered."""
    from flask import Flask
    from serving.api import api_bp, register_components
    from online.serving import RecommendationServer
    from core.config import get_config

    test_app = Flask(__name__)
    test_app.config['TESTING'] = True

    config = get_config()
    server = RecommendationServer(config)
    register_components(server=server, model_manager=None)
    test_app.register_blueprint(api_bp)
    return test_app


class TestServingAPIRecommend(unittest.TestCase):
    """Tests for POST /api/v1/recommend."""

    def setUp(self):
        self.app = _make_app()
        self.client = self.app.test_client()

    def test_recommend_valid_request_returns_200(self):
        """Valid recommend request returns HTTP 200."""
        response = self.client.post(
            '/api/v1/recommend',
            json={'user_id': 'test_user', 'result_size': 5},
        )
        self.assertEqual(response.status_code, 200)

    def test_recommend_without_user_id_returns_400(self):
        """Recommend request without user_id returns HTTP 400."""
        response = self.client.post('/api/v1/recommend', json={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

    def test_recommend_response_has_status(self):
        """Recommend response JSON contains a status field."""
        response = self.client.post(
            '/api/v1/recommend',
            json={'user_id': 'test_user'},
        )
        data = response.get_json()
        self.assertIn('status', data)

    def test_recommend_no_json_body_returns_400(self):
        """Recommend request with no JSON body returns HTTP 400."""
        response = self.client.post(
            '/api/v1/recommend',
            data='not-json',
            content_type='text/plain',
        )
        self.assertEqual(response.status_code, 400)


class TestServingAPIModels(unittest.TestCase):
    """Tests for GET /api/v1/models."""

    def setUp(self):
        self.app = _make_app()
        self.client = self.app.test_client()

    def test_list_models_returns_200(self):
        """GET /api/v1/models returns HTTP 200."""
        response = self.client.get('/api/v1/models')
        self.assertEqual(response.status_code, 200)

    def test_list_models_returns_loaded_versions(self):
        """GET /api/v1/models response contains loaded_versions key."""
        response = self.client.get('/api/v1/models')
        data = response.get_json()
        self.assertIn('loaded_versions', data)

    def test_list_models_without_manager_returns_empty_versions(self):
        """When no model_manager is set, loaded_versions is empty."""
        response = self.client.get('/api/v1/models')
        data = response.get_json()
        self.assertEqual(data['loaded_versions'], [])
        self.assertIsNone(data['active_version'])


class TestServingAPIModelReload(unittest.TestCase):
    """Tests for POST /api/v1/models/reload."""

    def setUp(self):
        self.app = _make_app()
        self.client = self.app.test_client()

    def test_reload_without_manager_returns_503(self):
        """Reload without model manager returns HTTP 503."""
        response = self.client.post(
            '/api/v1/models/reload',
            json={'version': 'v1', 'model_dir': '/tmp/model'},
        )
        self.assertEqual(response.status_code, 503)

    def test_reload_missing_fields_returns_400(self):
        """Reload request with missing required fields returns HTTP 400."""
        from serving.api import register_components
        from easyrec_extended.model_manager import ModelManager
        register_components(server=None, model_manager=ModelManager())

        response = self.client.post('/api/v1/models/reload', json={})
        self.assertEqual(response.status_code, 400)

    def tearDown(self):
        # Reset components so other tests are not affected
        from serving.api import register_components
        register_components(server=None, model_manager=None)


if __name__ == '__main__':
    unittest.main()
