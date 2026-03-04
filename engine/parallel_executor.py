"""
Parallel stage executor for the recommendation pipeline.

Provides :class:`ParallelStageExecutor` which runs multiple callables
concurrently via a shared thread pool and enforces per-callable timeouts.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Type alias: (name, result_or_None, elapsed_ms)
ExecutionResult = Tuple[str, Optional[Any], float]


class ParallelStageExecutor:
    """
    Executes multiple named callables in parallel using a reusable thread pool.

    Example::

        executor = ParallelStageExecutor(max_workers=8)
        results = executor.execute_parallel(
            [("engine_a", engine_a.recall, request),
             ("engine_b", engine_b.recall, request)],
            timeout_ms=500,
        )
        for name, result, elapsed in results:
            if result is not None:
                items.extend(result)
    """

    def __init__(self, max_workers: int = 8):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def execute_parallel(
        self,
        callables: List[Tuple[str, Callable, Any]],
        timeout_ms: float = 500.0,
    ) -> List[ExecutionResult]:
        """
        Run each callable in parallel and collect results within the timeout.

        Args:
            callables: List of ``(name, func, arg)`` tuples.  ``func`` is
                called as ``func(arg)``.
            timeout_ms: Maximum wall-clock time (milliseconds) to wait for
                each callable.  Callables that exceed this are cancelled and
                their result is ``None``.

        Returns:
            List of ``(name, result, elapsed_ms)`` tuples.  ``result`` is
            ``None`` when the callable timed out or raised an exception.
        """
        timeout_s = timeout_ms / 1000.0
        futures_map = {}
        start_times = {}

        for name, func, arg in callables:
            start_times[name] = time.monotonic()
            future = self._pool.submit(func, arg)
            futures_map[name] = future

        results: List[ExecutionResult] = []
        for name, future in futures_map.items():
            elapsed_ms = (time.monotonic() - start_times[name]) * 1000
            remaining = max(0.0, timeout_s - elapsed_ms / 1000.0)
            try:
                result = future.result(timeout=remaining)
                elapsed_ms = (time.monotonic() - start_times[name]) * 1000
                results.append((name, result, elapsed_ms))
            except FuturesTimeout:
                elapsed_ms = (time.monotonic() - start_times[name]) * 1000
                logger.warning(
                    f"ParallelStageExecutor: callable '{name}' timed out "
                    f"after {elapsed_ms:.1f}ms (limit={timeout_ms}ms)"
                )
                future.cancel()
                results.append((name, None, elapsed_ms))
            except Exception as exc:
                elapsed_ms = (time.monotonic() - start_times[name]) * 1000
                logger.warning(
                    f"ParallelStageExecutor: callable '{name}' raised "
                    f"{type(exc).__name__}: {exc}"
                )
                results.append((name, None, elapsed_ms))

        return results

    def shutdown(self, wait: bool = True):
        """Shut down the underlying thread pool."""
        self._pool.shutdown(wait=wait)
