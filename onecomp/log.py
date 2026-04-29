"""
OneComp logging utilities.

Provides a logging setup scoped to the onecomp namespace so that
importing OneComp never alters the application's root logger.

Design principles (aligned with vLLM / HuggingFace Transformers / PyTorch):

* Only the onecomp logger tree is configured — the root logger is
  never touched.
* ONECOMP_LOG_LEVEL env-var overrides the *level* argument.
* ONECOMP_CONFIGURE_LOGGING=0 disables all logging setup, allowing
  the host application to manage logging entirely.
* set_verbosity*() helpers can be called at any time to change the
  effective level without re-running setup_logger.
* warning_once / info_once deduplicate high-frequency messages.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

_LIBRARY_ROOT = "onecomp"

# ---------------------------------------------------------------------------
# tqdm control
# ---------------------------------------------------------------------------

_tqdm_disabled: bool | None = None


def should_disable_tqdm() -> bool:
    """Return True when tqdm progress bars should be suppressed.

    Resolution order:
    1. Explicit ``set_tqdm_disabled(flag)`` call
    2. ``ONECOMP_NO_PROGRESS=1`` env-var
    3. ``sys.stderr.isatty()`` — disable when not a terminal
    """
    if _tqdm_disabled is not None:
        return _tqdm_disabled
    if os.environ.get("ONECOMP_NO_PROGRESS") == "1":
        return True
    return not (hasattr(sys.stderr, "isatty") and sys.stderr.isatty())


def set_tqdm_disabled(disabled: bool) -> None:
    """Explicitly enable or disable tqdm progress bars globally."""
    global _tqdm_disabled  # noqa: PLW0603
    _tqdm_disabled = disabled


# ---------------------------------------------------------------------------
# Verbosity helpers (can be called any time)
# ---------------------------------------------------------------------------

def get_verbosity() -> int:
    """Return the current logging level of the ``onecomp`` logger."""
    return logging.getLogger(_LIBRARY_ROOT).getEffectiveLevel()


def set_verbosity(level: int) -> None:
    """Set the logging level for the ``onecomp`` logger.

    Args:
        level: A standard :mod:`logging` level constant
            (e.g. ``logging.DEBUG``, ``logging.INFO``).
    """
    logging.getLogger(_LIBRARY_ROOT).setLevel(level)


def set_verbosity_debug() -> None:
    """Shorthand for ``set_verbosity(logging.DEBUG)``."""
    set_verbosity(logging.DEBUG)


def set_verbosity_info() -> None:
    """Shorthand for ``set_verbosity(logging.INFO)``."""
    set_verbosity(logging.INFO)


def set_verbosity_warning() -> None:
    """Shorthand for ``set_verbosity(logging.WARNING)``."""
    set_verbosity(logging.WARNING)


def set_verbosity_error() -> None:
    """Shorthand for ``set_verbosity(logging.ERROR)``."""
    set_verbosity(logging.ERROR)


# ---------------------------------------------------------------------------
# Deduplicated logging
# ---------------------------------------------------------------------------

_seen_messages: set[str] = set()


def warning_once(logger: logging.Logger, msg: str, *args, **kwargs) -> None:
    """Log a warning message only the first time it is encountered.

    Useful for per-layer or per-module warnings that would otherwise
    flood the output during quantization loops.
    """
    key = f"{logger.name}:{msg}"
    if key not in _seen_messages:
        _seen_messages.add(key)
        logger.warning(msg, *args, **kwargs)


def info_once(logger: logging.Logger, msg: str, *args, **kwargs) -> None:
    """Log an info message only the first time it is encountered."""
    key = f"{logger.name}:{msg}"
    if key not in _seen_messages:
        _seen_messages.add(key)
        logger.info(msg, *args, **kwargs)


# ---------------------------------------------------------------------------
# Color detection
# ---------------------------------------------------------------------------

def _should_use_color(color: str, stream=None) -> bool:
    """Determine whether ANSI colors should be used.

    Priority:
    1. Explicit color parameter ("always" / "never")
    2. NO_COLOR env-var  (https://no-color.org/)
    3. ONECOMP_COLOR=1 env-var
    4. stream.isatty()
    """
    if color == "always":
        return True
    if color == "never":
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("ONECOMP_COLOR") == "1":
        return True
    if stream is None:
        stream = sys.stderr
    return hasattr(stream, "isatty") and stream.isatty()


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class OneCompFormatter(logging.Formatter):
    """Logging formatter with optional ANSI colors and multi-line alignment.

    Inspired by vLLM's ``NewLineFormatter`` (prefix alignment) and
    Cargo/Ruff (colored level names, clean layout).
    """

    _COLORS = {
        logging.DEBUG: "\033[90m",      # grey
        logging.INFO: "\033[32m",       # green
        logging.WARNING: "\033[33m",    # yellow
        logging.ERROR: "\033[31m",      # red
        logging.CRITICAL: "\033[31;1m", # bold red
    }
    _RESET = "\033[0m"

    def __init__(self, use_color: bool = False):
        super().__init__(fmt="%(levelname)-7s %(asctime)s %(message)s", datefmt="%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)

        if record.message and "\n" in record.message:
            prefix_end = msg.index(record.message)
            prefix = " " * prefix_end
            msg = msg.replace("\n", "\n" + prefix)

        if self.use_color:
            color = self._COLORS.get(record.levelno, "")
            levelname_end = msg.index(record.levelname) + len(record.levelname)
            colored_level = color + msg[:levelname_end] + self._RESET
            msg = colored_level + msg[levelname_end:]

        return msg


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no ANSI codes, includes date)."""

    def __init__(self):
        super().__init__(
            fmt="%(levelname)-7s %(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class _JSONFormatter(logging.Formatter):
    """JSON Lines formatter for structured/machine-readable logging.

    Each log record is emitted as a single JSON object per line,
    compatible with tools that consume NDJSON (e.g. ``jq``, Datadog,
    Elasticsearch).
    """

    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------------

_configured: bool = False


def setup_logger(
    level: str = "INFO",
    color: str = "auto",
    logfile: str | None = None,
    filemode: str = "w",
    log_format: str = "auto",
    disable_tqdm: bool | None = None,
    *,
    basic: str | None = None,
    **kwargs,
):
    """Configure the onecomp logger.

    Only the onecomp namespace is modified; the application's root
    logger is never touched.  Call this at most once — subsequent calls
    are no-ops (use set_verbosity to adjust the level later).

    Environment variables (take precedence over arguments):

    * ONECOMP_LOG_LEVEL: Override *level* (e.g. DEBUG).
    * ONECOMP_LOG_FORMAT: json for structured JSON Lines output.
    * ONECOMP_CONFIGURE_LOGGING=0: Skip all logging setup.
    * ONECOMP_NO_PROGRESS=1: Disable tqdm progress bars.
    * NO_COLOR: Disable ANSI colors (https://no-color.org/).
    * ONECOMP_COLOR=1: Force ANSI colors even without a TTY.

    Args:
        level: Logging level for the onecomp logger (default "INFO").
        color: Color mode — "auto" (default), "always", or "never".
        logfile: Optional path to a log file.
        filemode: File mode when writing to *logfile* (default "w").
        log_format: "auto" for human-readable output, "json" for
            JSON Lines.
        disable_tqdm: Explicitly enable (False) or disable (True)
            tqdm progress bars.  ``None`` (default) uses auto-detection.
        basic: **Deprecated** — formerly set the root logger level.
            Ignored since OneComp no longer modifies the root logger.
    """
    global _configured  # noqa: PLW0603

    if _configured:
        return
    _configured = True

    if os.environ.get("ONECOMP_CONFIGURE_LOGGING") == "0":
        return

    # Environment overrides
    env_level = os.environ.get("ONECOMP_LOG_LEVEL")
    if env_level:
        level = env_level.upper()

    env_format = os.environ.get("ONECOMP_LOG_FORMAT")
    if env_format:
        log_format = env_format.lower()

    if disable_tqdm is not None:
        set_tqdm_disabled(disable_tqdm)

    # Configure the library logger (NOT the root logger)
    lib_logger = logging.getLogger(_LIBRARY_ROOT)
    lib_logger.setLevel(level)

    if not lib_logger.handlers:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.NOTSET)

        if log_format == "json":
            console.setFormatter(_JSONFormatter())
        else:
            use_color = _should_use_color(color, stream=sys.stderr)
            console.setFormatter(OneCompFormatter(use_color=use_color))

        lib_logger.addHandler(console)

    # Prevent propagation to root so we never interfere with the
    # application's own logging configuration.
    lib_logger.propagate = False

    # Backward compat: logfile via kwargs
    actual_logfile = logfile or kwargs.get("logfile")
    if actual_logfile:
        file_handler = logging.FileHandler(actual_logfile, mode=filemode)
        file_handler.setLevel(logging.NOTSET)
        file_handler.setFormatter(_PlainFormatter())
        lib_logger.addHandler(file_handler)
