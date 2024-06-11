# Copyright (c) 2024 FhY Developers
# Christopher Priebe <cpriebe@ucsd.edu>
# Jason C Del Rio <j3delrio@ucsd.edu>
# Hadi S Esmaeilzadeh <hadi@ucsd.edu>
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be
# used to endorse or promote products derived from this software without specific prior
# written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""General utilities to construct a logger.

Functions:
    get_logger: creates a new logger with appropriate level and formatting.

"""

import logging
from pathlib import Path
from typing import Optional, Union

_default_format = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(funcName)s():%(lineno)d | %(message)s"
)


def get_logger(
    name: str,
    level: int = logging.DEBUG,
    stream: Optional[logging.Handler] = None,
    formatter: logging.Formatter = _default_format,
) -> logging.Logger:
    """Constructs a logger given a name and level.

    Args:
        name (str): logger name
        level (int): Valid logging level
        stream (logging.Handler): Output stream of logging. Defaults to streamHandler.
        formatter (logging.Formatter): Format of logging message

    Raises:
        TypeError: When an invalid stream type is provided.

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
    log: logging.Logger, path: Union[str, Path], level: int = logging.DEBUG
) -> None:
    """Append a FileHandler object to an existing logger object.

    Args:
        log (logging.Logger): Logger instance object
        path (str): valid file path to stream logs to
        level (int, optional): Logging level. Defaults to logging.DEBUG.

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
        log.setLevel(level)
        log.addHandler(handler)
        current = logging.getLevelName(log.level)
        log.debug(f'Modifying Log Level from "{previous}" to "{current}"')

    else:
        log.addHandler(handler)
