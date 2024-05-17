"""General utilities to construct a logger.

Functions:
    get_logger: creates a new logger with appropriate level and formatting.

"""

import logging
from typing import Optional

_default_format = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(funcName)s():%(lineno)d | %(message)s"
)


def get_logger(
    name: str,
    level: int = logging.DEBUG,
    stream: Optional[logging.Handler] = None,
    formatter: logging.Formatter = _default_format,
) -> logging.Logger:
    """Constructs a Logger given a name and level.

    Args:
        name (str): logger name
        level (int): Valid Logging Level
        stream (logging.Handler): Output Stream of logging. Defaults to StreamHandler.
        formatter (logging.Formatter): Format of Logging Message

    Raises:
        TypeError: When an invalid Stream Type is provided.

    """
    log: logging.Logger = logging.getLogger(name)
    log.setLevel(level)

    if stream is None:
        stream = logging.StreamHandler()

    elif not isinstance(stream, logging.Handler):
        raise TypeError(f"Invalid Stream Provided: {stream}")

    stream.setLevel(level)
    stream.setFormatter(formatter)
    log.addHandler(stream)

    return log


def add_file_handler(
    log: logging.Logger, path: str, level: int = logging.DEBUG
) -> None:
    """Append a FileHandler object to an existing logger object.

    Args:
        log (logging.Logger): Logger instance object
        path (str): valid file path to stream logs to
        level (int, optional): Logging Level. Defaults to logging.DEBUG.

    Note:
        This function may change the logging level of the log object itself. To emit
        a log message, the filterer first checks the log level compared to level of the
        message. Only if this passes, are the levels of the handlers evaluated. To
        ensure the appended file handler receives relevant messages, we update the
        logging level if appropriate.

    """
    handler = logging.FileHandler(path, mode="a")
    handler.setLevel(level)

    # Use previous formatting
    if log.hasHandlers():
        form = log.handlers[0].formatter
        handler.setFormatter(form)

    if level < log.level:
        previous = logging.getLevelName(log.level)
        current = logging.getLevelName(level)
        log.setLevel(level)
        log.addHandler(handler)
        log.debug(f"Modifying Log Level from `{previous}` to `{current}`")

    else:
        log.addHandler(handler)
