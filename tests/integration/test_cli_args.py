"""Test Behavior of various CLI Arguments."""

import os

import pytest
from fhy import __version__
from fhy.cli import Status

from .utils import access_cli


@pytest.fixture
def file_log():
    here = os.path.abspath(os.path.join(__file__, os.pardir))
    file = os.path.join(here, "sample.log")

    yield file

    # Remove temporary log file
    os.remove(file)
    assert not os.path.exists(file), "Expected File to be removed."


def test_version():
    code, response, _ = access_cli("--version")
    assert __version__ in response, "Expected Version to be Reported."
    assert "FhY" in response, "Expected Program Name to be Reported."
    assert code == Status.OK, "Expected Successful Response."


def test_no_file_error():
    """Test CLI without file argument raises errors and reports correctly to stderr."""
    code, _, error = access_cli("serialize", "--verbose")

    assert code == Status.USAGE_ERROR, "Expected to report User Error Status Code."
    assert "ERROR" in error, "Expected error message logged to stderr."


def test_file_exists_error():
    """Test that an invalid filepath raises errors as expected."""
    code, _, error = access_cli("serialize", "cthulhu.fhy")

    assert code == Status.USAGE_ERROR, "Expected to report User Error Status Code."
    assert "ERROR" in error, "Expected error message logged to stderr."
    assert "FileExistsError" in error, "Expected Mention of FileExistError."


def test_log_file(file_log):
    """Test that a logging file is created on user request."""
    # NOTE: This request will fail, but not before we create the log
    code, _, error = access_cli("serialize", "--log-file", file_log)

    assert os.path.exists(file_log), "Expected Log File to be created."
    assert "ERROR" in error, "Expected error message logged to stderr."

    with open(file_log) as f:
        text = f.read()

    assert "ERROR" in text, "Expected Error Message within file."
