"""Tests for ModelManager without requiring TensorFlow."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyrec_extended.model_manager import ModelManager


class TestModelManagerNoModel(unittest.TestCase):
    """ModelManager tests that do not require a real SavedModel."""

    def setUp(self):
        self.manager = ModelManager()

    def test_initial_state_not_ready(self):
        """A freshly created ModelManager is not ready."""
        self.assertFalse(self.manager.is_ready())

    def test_initial_loaded_versions_empty(self):
        """No versions are loaded initially."""
        self.assertEqual(self.manager.loaded_versions, [])

    def test_initial_active_version_is_none(self):
        """Active version is None when no model is loaded."""
        self.assertIsNone(self.manager.active_version)

    def test_predict_returns_empty_dict_when_no_model(self):
        """predict() returns empty dict when no model is loaded."""
        result = self.manager.predict({'feature': 1.0})
        self.assertEqual(result, {})

    def test_batch_predict_returns_empty_dicts_when_no_model(self):
        """batch_predict() returns list of empty dicts when no model loaded."""
        result = self.manager.batch_predict([{'feature': 1.0}, {'feature': 2.0}])
        self.assertEqual(result, [{}, {}])

    def test_set_active_version_raises_for_unknown_version(self):
        """set_active_version() raises ValueError for non-existent version."""
        with self.assertRaises(ValueError):
            self.manager.set_active_version('nonexistent')

    def test_reload_version_raises_for_unknown_version(self):
        """reload_version() raises ValueError for non-existent version."""
        with self.assertRaises(ValueError):
            self.manager.reload_version('nonexistent', '/tmp/model')

    def test_register_version_updates_loaded_versions(self):
        """Directly injecting a mock inference object updates loaded_versions."""
        from unittest.mock import MagicMock
        mock_inference = MagicMock()
        mock_inference.is_loaded = True
        self.manager._versions['v1'] = mock_inference
        self.manager._active_version = 'v1'
        self.assertIn('v1', self.manager.loaded_versions)

    def test_set_active_version_works_after_mock_load(self):
        """set_active_version() works when a version has been registered."""
        from unittest.mock import MagicMock
        mock_inference = MagicMock()
        mock_inference.is_loaded = True
        self.manager._versions['v1'] = mock_inference
        self.manager._active_version = 'v1'
        # Add a second version
        self.manager._versions['v2'] = mock_inference
        self.manager.set_active_version('v2')
        self.assertEqual(self.manager.active_version, 'v2')

    def test_is_ready_true_when_version_loaded(self):
        """is_ready() returns True when active version has is_loaded=True."""
        from unittest.mock import MagicMock
        mock_inference = MagicMock()
        mock_inference.is_loaded = True
        self.manager._versions['v1'] = mock_inference
        self.manager._active_version = 'v1'
        self.assertTrue(self.manager.is_ready())

    def test_is_ready_false_when_model_not_loaded(self):
        """is_ready() returns False when model is registered but not loaded."""
        from unittest.mock import MagicMock
        mock_inference = MagicMock()
        mock_inference.is_loaded = False
        self.manager._versions['v1'] = mock_inference
        self.manager._active_version = 'v1'
        self.assertFalse(self.manager.is_ready())


if __name__ == '__main__':
    unittest.main()
