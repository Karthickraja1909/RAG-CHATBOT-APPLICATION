"""
Centralized logging utility for GenAI RAG System.
Provides structured logging with configurable levels and formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import get_settings


def get_logger(
    name: str,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create and return a configured logger instance.

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

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger