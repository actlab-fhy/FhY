"""Tests Logger Utilities."""

import logging
import os
from typing import Callable, Generator, Optional

import pytest

from fhy.utils import logger


@pytest.fixture
def get_log() -> Generator[Callable[..., logging.Logger], None, None]:
    log: Optional[logging.Logger] = None

    def _inner(*args, **kwargs) -> logging.Logger:
        nonlocal log
        log = logger.get_logger(*args, **kwargs)
        return log

    yield _inner

    # Teardown Sequence Accounting for intentionally Failed Tests
    if log is not None:
        # A Hack to Remove the Temporary Logger made from Root Logging Manager Registry
        registry = logging.Logger.manager.loggerDict
        registry.pop(log.name)
        assert log.name not in registry, "Expected to Remove Logger from registry"

        # Remove Temporary Filepath of Filehandler Loggers
        if isinstance((handler := log.handlers[0]), logging.FileHandler):
            os.remove(handler.baseFilename)
            assert not os.path.exists(
                handler.baseFilename
            ), "Expected to Remove Temporary Logging Filepath."


@pytest.mark.parametrize(
    "name, level, stream",
    [
        ("test", logging.DEBUG, None),
        ("foo", logging.INFO, None),
        ("bar", logging.CRITICAL, None),
        ("baz", logging.WARNING, logging.StreamHandler()),
        ("baz", logging.WARNING, logging.FileHandler("test.log")),
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
