"""Package-level logger configuration."""

from __future__ import annotations

import logging
from typing import Final, Protocol


class LoggerLike(Protocol):
    """Minimal logger interface accepted by the package."""

    def debug(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log a debug-level message."""

    def info(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log an info-level message."""

    def error(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log an error-level message."""


_LOGGER_NAME = "mvp_dataset"
_DEFAULT_LOG_LEVEL: Final[int] = logging.INFO
_injected_logger: LoggerLike | None = None
_default_log_level = _DEFAULT_LOG_LEVEL


def _resolve_log_level(level: int | str) -> int:
    """Normalize a user-provided log level name or numeric value."""
    if isinstance(level, int):
        return level

    normalized = level.strip().upper()
    if not normalized:
        msg = "[InvalidLogLevel] log level must be a non-empty string or integer"
        raise ValueError(msg)

    if normalized.lstrip("-").isdigit():
        return int(normalized)

    level_value = logging.getLevelNamesMapping().get(normalized)
    if level_value is None:
        msg = f"[InvalidLogLevel] unsupported log level {level!r}"
        raise ValueError(msg)
    return level_value


def _default_logger() -> logging.Logger:
    """Return the package default logger, configuring it lazily once."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not getattr(logger, "_mvp_dataset_configured", False):
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.setLevel(_default_log_level)
        logger.propagate = False
        logger._mvp_dataset_configured = True
    return logger


def set_log_level(level: int | str) -> None:
    """Set the log level used by the package default logger.

    This only affects the built-in ``mvp_dataset`` logger returned by
    :func:`get_logger` when no custom logger has been injected with
    :func:`set_logger`.
    """
    global _default_log_level

    _default_log_level = _resolve_log_level(level)
    _default_logger().setLevel(_default_log_level)


def get_log_level() -> int:
    """Return the current log level of the package default logger."""
    return _default_logger().getEffectiveLevel()


def reset_log_level() -> None:
    """Restore the package default logger level to ``INFO``."""
    set_log_level(_DEFAULT_LOG_LEVEL)


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
