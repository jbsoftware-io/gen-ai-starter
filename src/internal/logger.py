"""
Structured JSON logging module with context support.

Provides a lightweight JSON logger similar to Winston, enabling
structured logging with thread-safe context management.
"""

import logging
import os
from contextvars import ContextVar
from typing import Any, Dict

from pythonjsonlogger import jsonlogger


# Thread-safe context storage
_log_context: ContextVar[Dict[str, Any]] = (
    ContextVar("log_context", default={})
)


def get_context() -> Dict[str, Any]:
    """Get the current logging context."""
    return _log_context.get().copy()


def set_context(
    key: str | Dict[str, Any], value: Any = None
) -> None:
    """
    Set context for logging.

    Args:
        key: Either a string key or a dict of key-value pairs
        value: The value to set (only used if key is a string)
    """
    context = _log_context.get().copy()

    if isinstance(key, dict):
        context.update(key)
    else:
        context[key] = value

    _log_context.set(context)


def clear_context() -> None:
    """Clear all logging context."""
    _log_context.set({})


class ContextJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that includes context in log records."""

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        """Add fields to log record, including context."""
        super().add_fields(log_record, record, message_dict)

        # Merge context into log record
        context = get_context()
        log_record.update(context)


def create_logger(name: str | None = None) -> logging.Logger:
    """
    Create a structured JSON logger.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        A configured logger instance
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = True

    return logger


def configure_root_logger() -> None:
    """
    Configure the root logger with JSON formatting.

    This ensures ALL loggers (including third-party ones) output JSON.
    Call this once at application startup.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []

    # Create console handler with JSON formatter
    handler = logging.StreamHandler()
    formatter = ContextJsonFormatter(
        fmt="%(timestamp)s %(level)s %(name)s %(message)s",
        timestamp=True,
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


# Configure root logger for JSON output globally
configure_root_logger()

# Global logger instance for direct use
logger = create_logger(__name__)
