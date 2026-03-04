"""Tests for FeatureService with dict-backed store."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyrec_extended.features.feature_service import FeatureService


class TestFeatureServiceDictStore(unittest.TestCase):
    """FeatureService tests using a plain dict as the backing store."""

    def setUp(self):
        self.store = {
            'user:user_1': {'age': '25', 'gender': 'M'},
            'item:item_1': {'category': 'electronics', 'price': '99.99'},
            'item:item_2': {'category': 'books', 'price': '12.50'},
        }
        self.service = FeatureService(feature_store=self.store)

    def test_get_user_features_returns_dict(self):
        """get_user_features() returns the feature dict for a known user."""
        features = self.service.get_user_features('user_1')
        self.assertEqual(features, {'age': '25', 'gender': 'M'})

    def test_get_user_features_unknown_user_returns_empty(self):
        """get_user_features() returns empty dict for unknown user."""
        features = self.service.get_user_features('unknown_user')
        self.assertEqual(features, {})

    def test_get_item_features_returns_dict(self):
        """get_item_features() returns feature dicts for known item ids."""
        result = self.service.get_item_features(['item_1', 'item_2'])
        self.assertEqual(result['item_1'], {'category': 'electronics', 'price': '99.99'})
        self.assertEqual(result['item_2'], {'category': 'books', 'price': '12.50'})

    def test_get_item_features_unknown_item_returns_empty(self):
        """get_item_features() returns empty dict for unknown item ids."""
        result = self.service.get_item_features(['unknown_item'])
        self.assertEqual(result['unknown_item'], {})

    def test_get_item_features_mixed_known_unknown(self):
        """get_item_features() handles a mix of known and unknown items."""
        result = self.service.get_item_features(['item_1', 'no_such_item'])
        self.assertIn('item_1', result)
        self.assertEqual(result['no_such_item'], {})

    def test_build_easyrec_features_merges_dicts(self):
        """build_easyrec_features() merges user, item, and context features."""
        user_feats = {'age': '25', 'gender': 'M'}
        item_feats = {'category': 'electronics', 'price': '99.99'}
        ctx_feats = {'request_time': '1234567890'}

        merged = self.service.build_easyrec_features(user_feats, item_feats, ctx_feats)

        self.assertEqual(merged['age'], '25')
        self.assertEqual(merged['category'], 'electronics')
        self.assertEqual(merged['request_time'], '1234567890')

    def test_build_easyrec_features_no_context(self):
        """build_easyrec_features() works without context features."""
        user_feats = {'age': '25'}
        item_feats = {'price': '9.99'}
        merged = self.service.build_easyrec_features(user_feats, item_feats)
        self.assertEqual(merged, {'age': '25', 'price': '9.99'})

    def test_build_easyrec_features_empty_inputs(self):
        """build_easyrec_features() returns empty dict for empty inputs."""
        merged = self.service.build_easyrec_features({}, {})
        self.assertEqual(merged, {})


class TestFeatureServiceEmptyStore(unittest.TestCase):
    """FeatureService tests with an empty dict store."""

    def setUp(self):
        self.service = FeatureService()

    def test_get_user_features_empty_store(self):
        self.assertEqual(self.service.get_user_features('any_user'), {})

    def test_get_item_features_empty_store(self):
        result = self.service.get_item_features(['item_1'])
        self.assertEqual(result['item_1'], {})


if __name__ == '__main__':
    unittest.main()
