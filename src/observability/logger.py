"""
Structured logging configuration using structlog.

Design decision: structlog over stdlib logging for three reasons:
1. JSON-structured output in production — machine-parseable by any log aggregator
2. Context binding — we can bind a trace_id once and it appears on every log line
   in that request without passing it explicitly through every function call
3. Human-readable console output in development — no trade-off needed

The logger is configured once at import time. All modules should use:
    from src.observability.logger import get_logger
    logger = get_logger()
"""

import sys
import logging
import structlog
from src.config import settings


def configure_logging() -> None:
    """
    Configure structlog processors and stdlib logging bridge.

    Called once at application startup. Subsequent calls are idempotent.
    """
    # Shared processors applied to every log event
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,  # Picks up trace_id from context
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # In production (INFO+), emit JSON for machine parsing.
    # In development (DEBUG), emit colorized console output.
    if settings.log_level.upper() == "DEBUG":
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging used by uvicorn and third-party clients through structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level.upper())

    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Return a bound structlog logger.

    Args:
        name: Optional logger name. If None, structlog infers from the call site.

    Returns:
        A BoundLogger that automatically includes trace_id from contextvars.
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


# Configure on import so logging is ready before any other module logs
configure_logging()
