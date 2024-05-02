"""Test the Identifier Collector AST Pass."""

from typing import Callable, Generator

import pytest

from fhy.ir import Identifier
from fhy.lang.ast import Module
from fhy.lang.passes.identifier_collector import (
    IdentifierCollector,
    collect_identifiers,
)


@pytest.fixture
def build_id() -> Generator[Callable[[str, int], Identifier], None, None]:
    """Build an Identifier Node, with a hacked ID."""

    def inner(name: str, value: int) -> Identifier:
        i = Identifier(name)
        i._id = value

        return i

    yield inner


@pytest.mark.parametrize("name", ["queen", "honey", "butter"])
@pytest.mark.parametrize("value", [7, 15, 27])
def test_collector_cls(build_id, name, value):
    """Test collection of Identifiers using the class directly on Identifier nodes."""
    node = build_id(name, value)
    instance = IdentifierCollector()
    instance(node)

    assert instance.identifiers == {node}, "Expected found Identifiers to Match."


@pytest.mark.parametrize("name", ["truffle", "pig"])
@pytest.mark.parametrize("value", [29, 121])
def test_collector_function(build_id, name, value):
    """Test Identifiers Collected by function api."""
    node = build_id(name, value)
    result = collect_identifiers(node)
    assert result == {node}, "Expected found Identifiers to Match."


def test_empty_module():
    """Test an empty module returns empty set."""
    collect_identifiers(Module()) == set()
