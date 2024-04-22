""" """

from dataclasses import dataclass
from typing import Optional

import pytest

from fhy.lang.ast import directory
from fhy.lang.ast.base import ASTNode
from fhy.lang.span import Span


@pytest.fixture
def generic_node():
    @dataclass(frozen=True, kw_only=True)
    class __GenericTestNode(ASTNode):
        x: int
        y: float

    return __GenericTestNode


@pytest.fixture
def bad_subclass_node(generic_node):
    @dataclass(frozen=True, kw_only=True)
    class __SubclassedTestNode(generic_node):
        x: float
        z: float

    return __SubclassedTestNode


@pytest.fixture
def setup_ast_node(request):
    """Registers an AST Node and Removes Node from Registry on Teardown"""
    # Dynamic Fixture Argument (of another Fixture)
    node = request.getfixturevalue(request.param)
    yield directory.register_ast_node(node)

    # Teardown
    directory._ast_node_types.pop(node)
    assert (
        node not in directory._ast_node_types
    ), "Fixture Teardown Incomplete. Node Still Registered"


@pytest.mark.xfail(
    reason="Retrieving an Unregistered ASTNode", raises=directory.UnregisteredASTNode
)
def test_unregistered_node_error():
    """Tests that retrieiving an unregistered class raises
    an UnregisteredASTNode Exception.

    """
    directory.get_ast_node_type_info(int)


@pytest.mark.parametrize("setup_ast_node", ["generic_node"], indirect=True)
def test_register_ast_node(setup_ast_node):
    """Confirms we Correctly Register a New ASTNode Class"""
    node = setup_ast_node

    assert issubclass(
        node, ASTNode
    ), "Expected __GenericTestNode to be a Sublcass of ASTNode"
    assert (
        node in directory._ast_node_types
    ), "Expected __GenericTestNode to be Registered"

    ret = directory.get_ast_node_type_info(node)
    expected = {"span": Optional[Span], "x": int, "y": float}
    assert ret.fields == expected, "Expected Correct Annotations of __GenericTestNode"


@pytest.mark.parametrize("setup_ast_node", ["bad_subclass_node"], indirect=True)
def test_register_ast_node_with_bad_subclassing(setup_ast_node):
    node = setup_ast_node

    assert issubclass(
        node, ASTNode
    ), "Expected __SubclassedTestNode to be a Sublcass of ASTNode"
    assert (
        node in directory._ast_node_types
    ), "Expected __SubclassedTestNode to be Registered"

    ret = directory.get_ast_node_type_info(node)
    expected = {"span": Optional[Span], "x": float, "y": float, "z": float}
    assert ret.fields == expected, "Expected Correct Annotation"
