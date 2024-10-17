"""Test the Identifier Collector AST Pass."""

from collections.abc import Callable, Generator

import pytest
from fhy.ir import CoreDataType, Identifier, NumericalType, TypeQualifier
from fhy.lang.ast import (
    DeclarationStatement,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
)
from fhy.lang.ast.passes.identifier_collector import (
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


def _qualified_type():
    return QualifiedType(
        base_type=NumericalType(data_type=CoreDataType.INT32, shape=[]),
        type_qualifier=TypeQualifier.INPUT,
    )


def test_declaration_statement(build_id):
    """Test retrieval of ID from Declaration Statement."""
    identity = build_id("rosanne", 23)

    statement = DeclarationStatement(
        variable_name=identity,
        variable_type=_qualified_type(),
        expression=IntLiteral(value=5),
    )

    result = collect_identifiers(statement)
    assert result == {identity}, "Expected to retrieve one Identifier"


def test_empty_procedure(build_id):
    """Test retrieval of ID from Empty Procedure Component."""
    identity = build_id("bar", 96)

    proc = Procedure(name=identity, args=[], body=[])

    result = collect_identifiers(proc)
    assert result == {identity}, "Expected to retrieve one Identifier"


def test_empty_operation(build_id):
    """Test retrieval of ID from Empty Operation Component."""
    identity = build_id("foo", 117)

    op = Operation(name=identity, args=[], body=[], return_type=_qualified_type())

    result = collect_identifiers(op)
    assert result == {identity}, "Expected to retrieve one Identifier"
