"""Tests for structured logging configuration."""
import sys
import os
import logging
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyrec_extended.logging_config import configure_logging, RequestContext


class TestRequestContext(unittest.TestCase):
    """Tests for RequestContext context-variable bag."""

    def setUp(self):
        RequestContext.clear()

    def tearDown(self):
        RequestContext.clear()

    def test_get_returns_empty_dict_initially(self):
        """get() returns an empty dict before any values are set."""
        self.assertEqual(RequestContext.get(), {})

    def test_set_stores_values(self):
        """set() stores arbitrary key-value pairs."""
        RequestContext.set(request_id='abc', user_id='user_1')
        ctx = RequestContext.get()
        self.assertEqual(ctx['request_id'], 'abc')
        self.assertEqual(ctx['user_id'], 'user_1')

    def test_set_merges_values(self):
        """Multiple set() calls merge values."""
        RequestContext.set(request_id='abc')
        RequestContext.set(user_id='user_1')
        ctx = RequestContext.get()
        self.assertIn('request_id', ctx)
        self.assertIn('user_id', ctx)

    def test_clear_resets_context(self):
        """clear() removes all stored values."""
        RequestContext.set(request_id='abc')
        RequestContext.clear()
        self.assertEqual(RequestContext.get(), {})

    def test_get_value_returns_default_for_missing_key(self):
        """get_value() returns default for a key that was not set."""
        result = RequestContext.get_value('nonexistent', default='fallback')
        self.assertEqual(result, 'fallback')

    def test_get_value_returns_stored_value(self):
        """get_value() returns the stored value."""
        RequestContext.set(stage='recall')
        self.assertEqual(RequestContext.get_value('stage'), 'recall')

    def test_get_returns_copy(self):
        """get() returns a copy, not the internal dict."""
        RequestContext.set(key='val')
        copy = RequestContext.get()
        copy['extra'] = 'injected'
        self.assertNotIn('extra', RequestContext.get())


class TestConfigureLogging(unittest.TestCase):
    """Tests for configure_logging()."""

    def test_configure_logging_sets_root_level(self):
        """configure_logging() sets the root logger level."""
        configure_logging('DEBUG')
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_configure_logging_default_level_info(self):
        """configure_logging() with no arg uses INFO by default."""
        os.environ.pop('LOG_LEVEL', None)
        configure_logging()
        self.assertEqual(logging.getLogger().level, logging.INFO)

    def test_configure_logging_env_var(self):
        """configure_logging() respects LOG_LEVEL env var."""
        os.environ['LOG_LEVEL'] = 'WARNING'
        configure_logging()
        self.assertEqual(logging.getLogger().level, logging.WARNING)
        del os.environ['LOG_LEVEL']

    def test_configure_logging_adds_handler(self):
        """configure_logging() attaches at least one handler to root logger."""
        configure_logging()
        self.assertGreater(len(logging.getLogger().handlers), 0)

    def test_configure_logging_handler_has_json_formatter(self):
        """configure_logging() uses JSON formatter."""
        from pythonjsonlogger import jsonlogger
        configure_logging()
        root = logging.getLogger()
        formatters = [h.formatter for h in root.handlers]
        self.assertTrue(
            any(isinstance(f, jsonlogger.JsonFormatter) for f in formatters),
            "Expected at least one JsonFormatter handler",
        )


if __name__ == '__main__':
    unittest.main()
