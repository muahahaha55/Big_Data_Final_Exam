"""Structured logging configuration.

Uses structlog for machine-parseable JSON logs in production and
human-readable colored logs in dev. Critical for observability when
running in containers or AWS Lambda.

Usage:
    from credit_risk.utils.logging import get_logger
    log = get_logger(__name__)
    log.info("scoring_request", customer_id="123", pd=0.087, latency_ms=42)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    json_logs: bool | None = None,
) -> None:
    """Configure both stdlib logging and structlog.

    Args:
        level: Log level (DEBUG/INFO/WARNING/ERROR).
        json_logs: Force JSON output. Default: True if not a TTY (i.e. in
            containers/CI/Lambda), False if interactive.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if json_logs is None:
        json_logs = not sys.stderr.isatty()

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=log_level, stream=sys.stderr)

    # Silence noisy third-party loggers
    for noisy in ("py4j", "pyspark", "urllib3", "botocore", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger. Configures logging on first call."""
    if not _CONFIGURED:
        configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind context vars that appear in every subsequent log line.

    Example: bind_context(request_id="abc", customer_id="123")
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context vars (call at request end)."""
    structlog.contextvars.clear_contextvars()
