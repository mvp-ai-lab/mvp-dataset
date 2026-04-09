"""Package-level logger configuration."""

from __future__ import annotations

import logging
from typing import Protocol


class LoggerLike(Protocol):
    """Minimal logger interface accepted by the package."""

    def debug(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log a debug-level message."""

    def info(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log an info-level message."""

    def error(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log an error-level message."""


_LOGGER_NAME = "mvp_dataset"
_injected_logger: LoggerLike | None = None


def _default_logger() -> logging.Logger:
    """Return the package default logger, configuring it lazily once."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def set_logger(logger: LoggerLike) -> None:
    """Install a process-global logger used by package subsystems."""
    global _injected_logger
    _injected_logger = logger


def reset_logger() -> None:
    """Clear the injected logger and fall back to the package default logger."""
    global _injected_logger
    _injected_logger = None


def get_logger() -> LoggerLike:
    """Return the injected logger, or the package default logger."""
    if _injected_logger is not None:
        return _injected_logger
    return _default_logger()
