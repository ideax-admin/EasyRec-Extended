"""
Structured JSON logging configuration for EasyRec-Extended.

Configure application-wide structured logging::

    from easyrec_extended.logging_config import configure_logging
    configure_logging()

Use the request-scoped context to attach metadata to all log records
emitted during a request::

    from easyrec_extended.logging_config import RequestContext
    RequestContext.set(request_id="abc-123", user_id="user_42")
    # All subsequent log calls in this thread / async task will include
    # request_id and user_id fields automatically.
"""
import logging
import os
from contextvars import ContextVar
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger

# ---------------------------------------------------------------------------
# Request-scoped context
# ---------------------------------------------------------------------------

_request_context: ContextVar[Dict[str, Any]] = ContextVar(
    "_request_context", default={}
)


class RequestContext:
    """Thread / async-task local bag of request-scoped metadata.

    Metadata stored here is automatically injected into every log record by
    :class:`RequestContextFilter`.

    Example::

        RequestContext.set(request_id="abc", user_id="u1", stage="recall")
        RequestContext.clear()
    """

    @staticmethod
    def set(**kwargs: Any):
        """Merge the given key-value pairs into the current context.

        Args:
            **kwargs: Arbitrary key-value pairs to store (e.g. request_id, user_id).
        """
        ctx = dict(_request_context.get())
        ctx.update(kwargs)
        _request_context.set(ctx)

    @staticmethod
    def get() -> Dict[str, Any]:
        """Return a copy of the current context dict."""
        return dict(_request_context.get())

    @staticmethod
    def clear():
        """Reset the context for the current task/thread."""
        _request_context.set({})

    @staticmethod
    def get_value(key: str, default: Any = None) -> Any:
        """Retrieve a single value from the current context.

        Args:
            key: The context key to look up.
            default: Value returned when the key is absent.

        Returns:
            The stored value or *default*.
        """
        return _request_context.get().get(key, default)


# ---------------------------------------------------------------------------
# Logging filter that injects request context into every log record
# ---------------------------------------------------------------------------

class RequestContextFilter(logging.Filter):
    """Logging :class:`logging.Filter` that injects :class:`RequestContext` data.

    Attach to any handler or logger to automatically include ``request_id``,
    ``user_id``, ``stage``, etc. in every emitted record.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        """Inject context metadata into *record*.

        Args:
            record: The log record being processed.

        Returns:
            Always ``True`` (record is never suppressed).
        """
        ctx = RequestContext.get()
        for key, value in ctx.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


# ---------------------------------------------------------------------------
# Public configuration entry-point
# ---------------------------------------------------------------------------

def configure_logging(log_level: Optional[str] = None):
    """Configure root logger to emit structured JSON records.

    The log level is resolved in the following order:
    1. The *log_level* argument (if provided).
    2. The ``LOG_LEVEL`` environment variable.
    3. ``INFO`` as the default.

    Args:
        log_level: Optional override for the log level string
            (e.g. ``"DEBUG"``, ``"WARNING"``).
    """
    resolved_level = (
        log_level
        or os.environ.get("LOG_LEVEL", "INFO")
    ).upper()

    numeric_level = getattr(logging, resolved_level, logging.INFO)

    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    ctx_filter = RequestContextFilter()

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(ctx_filter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    logging.getLogger(__name__).info(
        "Structured JSON logging configured", extra={"log_level": resolved_level}
    )
