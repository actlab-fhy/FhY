"""Tests Logger Utilities."""

import logging
import os
from collections.abc import Callable, Generator
from pathlib import Path
from typing import cast

import pytest
from fhy import logger


@pytest.fixture
def get_log() -> Generator[Callable[..., logging.Logger], None, None]:
    log: logging.Logger | None = None

    def _inner(*args, **kwargs) -> logging.Logger:
        nonlocal log
        log = logger.get_logger(*args, **kwargs)

        return log

    yield _inner

    # Teardown Sequence Accounting for intentionally Failed Tests
    # https://github.com/microsoft/pylance-release/issues/1355
    log = cast(logging.Logger, log) if log is not None else None
    if log is None:
        return None

    # A Hack to Remove the Temporary Logger made from Root Logging Manager Registry
    registry = logging.Logger.manager.loggerDict
    registry.pop(log.name)
    assert log.name not in registry, "Expected to Remove Logger from registry"

    def _test(x) -> bool:
        return isinstance(x, logging.FileHandler)

    file_handlers = filter(_test, log.handlers[:])
    log.handlers.clear()
    log.propagate = False

    for handler in file_handlers:
        filename = handler.baseFilename

        # Flush to stream to ensure all messages have been processed and closed.
        handler.flush()
        handler.close()

        # Remove the created log file
        os.remove(filename)
        assert not os.path.exists(filename), "Expected to Remove Logging Filepath."


def _assert_logger(log, name, level, stream):
    assert log.name == name, "Incorrect Assigned Logger Name"
    assert log.level == level, "Incorrect Logging Level Assigned"
    assert log.handlers[0].level == level, "Incorrect Logging Level Assigned to Stream"
    assert (
        log.handlers[0].formatter == logger._default_format
    ), "Expected Default Formatter"

    if stream is not None:
        assert log.handlers[0] == stream, "Unexpected Handler"
    else:
        assert isinstance(
            log.handlers[0], logging.StreamHandler
        ), "Expected StreamHandler Instance"


@pytest.mark.parametrize(
    "name, level, stream",
    [
        ("test", logging.DEBUG, None),
        ("foo", logging.INFO, None),
        ("bar", logging.CRITICAL, None),
        ("baz", logging.WARNING, logging.StreamHandler()),
    ],
)
def test_build_logger(get_log, name, level, stream):
    """Confirm We construct a Logger as expected."""
    log = get_log(name, level, stream)
    _assert_logger(log, name, level, stream)


def test_build_logger_with_file_handler(get_log):
    # NOTE: We separate this test, because teardown is not compatible on windows systems
    #       when it is parametrized, due to use of pytest-xdist, where multiple
    #       processes maintain an open stream to the file. This causes a PermissionError
    #       to occur, when we attempt to remove the created log file.
    name, level, stream = (
        "bun",
        logging.WARNING,
        logging.FileHandler("test_build_logger.log"),
    )
    log = get_log(name, level, stream)
    _assert_logger(log, name, level, stream)


def test_invalid_stream(get_log):
    """Confirm we raise a TypeError when providing an invalid stream arg."""
    with pytest.raises(TypeError):
        log = get_log("snap", logging.WARNING, "InvalidStream")


def test_append_file_handler(get_log):
    """Confirm a File Handler is appended to existing log."""
    log = get_log("name", logging.INFO)
    assert len(log.handlers) == 1, "Expected a Single Handler."

    pathname = str(Path("sample.log").resolve())
    logger.add_file_handler(log, pathname, logging.DEBUG)
    assert len(log.handlers) == 2, "Expected an Additional Handler."
    assert log.level == logging.DEBUG, "Expected Log Level to be Modified."

    handler = log.handlers[1]
    assert isinstance(handler, logging.FileHandler), "Expected new FileHandler"
    assert handler.baseFilename == pathname, "Expected Same Log Path Name."
