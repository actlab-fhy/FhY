"""Test IO Methods."""

import os
import pathlib

import pytest

from fhy.driver import file_reader

HERE = os.path.abspath(os.path.join(__file__, os.pardir))


def test_file_exists_error():
    """Confirm an Error is raised when provided with Invalid Filepath."""
    path = pathlib.Path(os.path.join(HERE, "invalid_file.fhy"))

    with pytest.raises(FileExistsError):
        file_reader.read_file(path)
