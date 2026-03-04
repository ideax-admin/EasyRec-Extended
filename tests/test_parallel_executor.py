"""Tests for engine/parallel_executor.py."""
import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.parallel_executor import ParallelStageExecutor


def _fast(arg):
    return [f"item_{arg}"]


def _slow(arg):
    time.sleep(2)
    return [f"slow_item_{arg}"]


def _raise(arg):
    raise RuntimeError("deliberate error")


class TestParallelStageExecutor(unittest.TestCase):

    def setUp(self):
        self.executor = ParallelStageExecutor(max_workers=4)

    def tearDown(self):
        self.executor.shutdown(wait=False)

    def test_all_complete_within_timeout(self):
        """All fast callables return results."""
        results = self.executor.execute_parallel(
            [('a', _fast, 'x'), ('b', _fast, 'y')],
            timeout_ms=5000,
        )
        self.assertEqual(len(results), 2)
        names = {r[0] for r in results}
        self.assertIn('a', names)
        self.assertIn('b', names)
        for name, result, elapsed in results:
            self.assertIsNotNone(result)

    def test_timeout_callable_returns_none(self):
        """A callable that exceeds the timeout produces a None result."""
        results = self.executor.execute_parallel(
            [('slow', _slow, 'z')],
            timeout_ms=100,
        )
        self.assertEqual(len(results), 1)
        name, result, elapsed = results[0]
        self.assertEqual(name, 'slow')
        self.assertIsNone(result)

    def test_exception_returns_none(self):
        """A callable that raises an exception produces a None result."""
        results = self.executor.execute_parallel(
            [('bad', _raise, 'anything')],
            timeout_ms=5000,
        )
        self.assertEqual(len(results), 1)
        name, result, elapsed = results[0]
        self.assertEqual(name, 'bad')
        self.assertIsNone(result)

    def test_mixed_results(self):
        """Fast callables succeed while slow/raising ones return None."""
        results = self.executor.execute_parallel(
            [('good', _fast, 'val'), ('bad', _raise, 'x')],
            timeout_ms=5000,
        )
        result_map = {r[0]: r[1] for r in results}
        self.assertIsNotNone(result_map['good'])
        self.assertIsNone(result_map['bad'])

    def test_all_fail_returns_all_none(self):
        """When all callables fail, all results are None."""
        results = self.executor.execute_parallel(
            [('e1', _raise, 'a'), ('e2', _raise, 'b')],
            timeout_ms=5000,
        )
        for _, result, _ in results:
            self.assertIsNone(result)

    def test_empty_callables(self):
        """Empty callables list returns empty results."""
        results = self.executor.execute_parallel([], timeout_ms=500)
        self.assertEqual(results, [])

    def test_elapsed_ms_is_non_negative(self):
        """Elapsed time reported for each callable is non-negative."""
        results = self.executor.execute_parallel(
            [('a', _fast, 'v')], timeout_ms=5000
        )
        for _, _, elapsed in results:
            self.assertGreaterEqual(elapsed, 0)


if __name__ == '__main__':
    unittest.main()
