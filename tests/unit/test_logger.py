"""Tests Logger Utilities."""

import logging
import os
from pathlib import Path
from typing import Callable, Generator, Optional

import pytest

from fhy import logger


@pytest.fixture
def get_log() -> Generator[Callable[..., logging.Logger], None, None]:
    log: Optional[logging.Logger] = None

    def _inner(*args, **kwargs) -> logging.Logger:
        nonlocal log
        log = logger.get_logger(*args, **kwargs)
        return log

    yield _inner

    # Teardown Sequence Accounting for intentionally Failed Tests
    if log is None:
        return None

    # A Hack to Remove the Temporary Logger made from Root Logging Manager Registry
    registry = logging.Logger.manager.loggerDict
    registry.pop(log.name)
    assert log.name not in registry, "Expected to Remove Logger from registry"

    # Remove Temporary Filepath of Filehandler Loggers
    def _test(x) -> bool:
        return isinstance(x, logging.FileHandler)

    if not any(map(_test, log.handlers)):
        return None

    for hand in log.handlers:
        if not _test(hand):
            continue
        os.remove(hand.baseFilename)
        assert not os.path.exists(
            hand.baseFilename
        ), "Expected to Remove Temporary Logging Filepath."


@pytest.mark.parametrize(
    "name, level, stream",
    [
        ("test", logging.DEBUG, None),
        ("foo", logging.INFO, None),
        ("bar", logging.CRITICAL, None),
        ("baz", logging.WARNING, logging.StreamHandler()),
        ("bun", logging.WARNING, logging.FileHandler("test.log")),
    ],
)
def test_build_logger(get_log, name, level, stream):
    """Confirm We construct a Logger as expected."""
    log = get_log(name, level, stream)

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
