"""
Centralized logging utility for GenAI RAG System.
Provides structured logging with correlation IDs for end-to-end request tracing.
"""

import contextvars
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

from config.settings import get_settings

# ─── Correlation ID Context ──────────────────────────────────────
# Thread/async-safe correlation ID that propagates across all modules
# for a single request. Set once at request entry, read everywhere.
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set correlation ID for the current request context. Returns the ID."""
    cid = cid or uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the current correlation ID (empty string if not set)."""
    return _correlation_id.get()


class _StructuredFormatter(logging.Formatter):
    """Formatter that injects correlation_id into every log record."""

    def __init__(self, fmt: str, datefmt: str, json_output: bool = False):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._json_output = json_output

    def format(self, record: logging.LogRecord) -> str:
        record.correlation_id = _correlation_id.get() or "-"
        if self._json_output:
            log_entry = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "correlation_id": record.correlation_id,
                "module": record.name,
                "function": record.funcName,
                "line": record.lineno,
                "message": record.getMessage(),
            }
            if record.exc_info and record.exc_info[0]:
                log_entry["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_entry, default=str)
        return super().format(record)


def get_logger(
    name: str,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create and return a configured logger instance with correlation ID support.

    Args:
        name: Logger name (typically __name__ of the calling module).
        log_file: Optional file path for log output.

    Returns:
        Configured logger instance.
    """
    settings = get_settings()

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    console_fmt = (
        "%(asctime)s | %(levelname)-8s | %(correlation_id)s | "
        "%(name)s | %(funcName)s:%(lineno)d | %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Console handler — human-readable
    console_formatter = _StructuredFormatter(fmt=console_fmt, datefmt=datefmt, json_output=False)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler — JSON structured (if configured)
    log_target = log_file or settings.log_file
    if log_target:
        log_path = Path(log_target)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        json_formatter = _StructuredFormatter(fmt=console_fmt, datefmt=datefmt, json_output=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger