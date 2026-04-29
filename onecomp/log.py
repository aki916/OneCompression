"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import logging
import os
import sys


def _should_use_color(color: str, stream=None) -> bool:
    """Determine whether ANSI colors should be used.

    Priority:
    1. Explicit ``color`` parameter ("always" / "never")
    2. ``NO_COLOR`` env-var  (https://no-color.org/)
    3. ``ONECOMP_COLOR=1`` env-var
    4. ``stream.isatty()``
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


def setup_logger(
    level: str = "INFO",
    basic: str = "WARNING",
    filemode: str = "w",
    color: str = "auto",
    **kwargs,
):
    """Setup the logger.

    Args:
        level: Logging level for the ``onecomp`` logger.
        basic: Logging level for the root logger used in ``basicConfig``.
        filemode: File mode when writing to a log file.
        color: Color mode for console output.
            ``"auto"`` (default) enables colors when stderr is a terminal.
            ``"always"`` / ``"never"`` force colors on or off.
        **kwargs: Additional keyword arguments.
        logfile: Name of the log file to write logs to.
    """
    root = logging.getLogger()
    root.setLevel(basic)

    if not root.handlers:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(basic)
        use_color = _should_use_color(color, stream=sys.stderr)
        console.setFormatter(OneCompFormatter(use_color=use_color))
        root.addHandler(console)
    else:
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                use_color = _should_use_color(color, stream=handler.stream)
                handler.setFormatter(OneCompFormatter(use_color=use_color))
                break

    if "logfile" in kwargs:
        file_handler = logging.FileHandler(kwargs["logfile"], mode=filemode)
        file_handler.setLevel(basic)
        file_handler.setFormatter(_PlainFormatter())
        root.addHandler(file_handler)

    logging.getLogger("onecomp").setLevel(level)
