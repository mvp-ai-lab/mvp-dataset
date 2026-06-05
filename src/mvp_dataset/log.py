"""Package-level logger configuration."""

from __future__ import annotations

import logging
from typing import Final, Protocol


class LoggerLike(Protocol):
    """Minimal logger interface accepted by the package."""

    def debug(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log a debug message.

        Args:
            msg: Message object passed to the logger.
            args: Positional arguments forwarded to the source constructor.
            kwargs: Keyword arguments forwarded to the source constructor.

        Returns:
            The result of the operation."""

    def info(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log an info message.

        Args:
            msg: Message object passed to the logger.
            args: Positional arguments forwarded to the source constructor.
            kwargs: Keyword arguments forwarded to the source constructor.

        Returns:
            The result of the operation."""

    def warning(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log a warning message.

        Args:
            msg: Message object passed to the logger.
            args: Positional arguments forwarded to the source constructor.
            kwargs: Keyword arguments forwarded to the source constructor.

        Returns:
            The result of the operation."""

    def error(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log an error message.

        Args:
            msg: Message object passed to the logger.
            args: Positional arguments forwarded to the source constructor.
            kwargs: Keyword arguments forwarded to the source constructor.

        Returns:
            The result of the operation."""


_LOGGER_NAME = "mvp_dataset"
_DEFAULT_LOG_LEVEL: Final[int] = logging.INFO
_injected_logger: LoggerLike | None = None
_default_log_level = _DEFAULT_LOG_LEVEL


def _format_message(msg: object, args: tuple[object, ...]) -> object:
    """Render stdlib-style log arguments into a single message."""
    if not args:
        return msg

    rendered = str(msg)
    try:
        return rendered % args
    except Exception:
        return " ".join((rendered, *(str(arg) for arg in args)))


class _InjectedLoggerAdapter:
    """Normalize injected loggers to the stdlib logging call shape."""

    def __init__(self, logger: LoggerLike) -> None:
        """Initialize the object."""
        self._logger = logger

    def _call(self, level: str, msg: object, *args: object, **kwargs: object) -> object:
        """Forward a log record to the wrapped logger function."""
        method = getattr(self._logger, level)
        rendered = _format_message(msg, args)
        if kwargs:
            try:
                return method(rendered, **kwargs)
            except TypeError:
                pass
        return method(rendered)

    def debug(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log a debug-level message."""
        return self._call("debug", msg, *args, **kwargs)

    def info(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log an info-level message."""
        return self._call("info", msg, *args, **kwargs)

    def warning(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log a warning-level message."""
        return self._call("warning", msg, *args, **kwargs)

    def error(self, msg: object, *args: object, **kwargs: object) -> object:
        """Log an error-level message."""
        return self._call("error", msg, *args, **kwargs)


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
    """Set the package log level.

    Args:
        level: Log level name or numeric value.

    Returns:
        None."""
    global _default_log_level

    _default_log_level = _resolve_log_level(level)
    _default_logger().setLevel(_default_log_level)


def get_log_level() -> int:
    """Return the active package log level.

    Returns:
        The active numeric log level."""
    return _default_logger().getEffectiveLevel()


def reset_log_level() -> None:
    """Restore the default package log level.

    Returns:
        None."""
    set_log_level(_DEFAULT_LOG_LEVEL)


def set_logger(logger: LoggerLike) -> None:
    """Replace the package logger.

    Args:
        logger: Logger-like object that receives log messages.

    Returns:
        None."""
    global _injected_logger
    if isinstance(logger, _InjectedLoggerAdapter):
        _injected_logger = logger
    else:
        _injected_logger = _InjectedLoggerAdapter(logger)


def reset_logger() -> None:
    """Restore the default package logger.

    Returns:
        None."""
    global _injected_logger
    _injected_logger = None


def get_logger() -> LoggerLike:
    """Return the active package logger.

    Returns:
        The active logger-like object."""
    if _injected_logger is not None:
        return _injected_logger
    return _default_logger()
