"""Structured logging with request-id correlation."""

import logging
import sys
from contextvars import ContextVar

import structlog

# Context var carries request_id through async call stack
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def _add_request_id(_logger: object, _method: str, event_dict: dict) -> dict:
    event_dict["request_id"] = request_id_ctx.get()
    return event_dict


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog + stdlib logging.

    JSON output by default — parseable by log aggregators.
    In local dev with `LOG_JSON=false` you get human-readable output.
    """
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_request_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Silence uvicorn access log noise — we have our own middleware
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
