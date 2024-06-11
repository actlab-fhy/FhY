"""Pytest Unit Test Fixtures and Utilities.

Fixtures present in this file are innately available to all modules present within this
directory. This is not true of Subdirectories, which will need it's own conftest.py
file.

"""

from typing import Callable

import pytest
from fhy.lang.ast import ASTNode
from fhy.lang.converter.from_fhy_source import from_fhy_source as fhy_source
from fhy.logger import get_logger

log = get_logger(__name__, 10)


@pytest.fixture
def construct_ast() -> Callable[[str], ASTNode]:
    """Construct an Abstract Syntax Tree (AST) from a raw text file source."""

    def _inner(source: str) -> ASTNode:
        return fhy_source(source, log=log)

    return _inner
