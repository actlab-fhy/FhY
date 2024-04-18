""" """
from dataclasses import dataclass
import pytest

from fhy.lang.ast.base import ASTNode
from fhy.lang.ast import directory
from fhy.lang.span import Span


@pytest.fixture
def setup_ast_node(request):
    @directory.register_ast_node
    @dataclass(frozen=True, kw_only=True)
    class __GenericTestNode(ASTNode):
        x: int
        y: float

    def teardown():
        directory._ast_node_types.pop(__GenericTestNode)

    request.addfinalizer(teardown)

    return __GenericTestNode


@pytest.fixture
def setup_bad_ast_nodes(request):
    @directory.register_ast_node
    @dataclass(frozen=True, kw_only=True)
    class __GenericTestNode(ASTNode):
        x: int
        y: float


    @directory.register_ast_node
    @dataclass(frozen=True, kw_only=True)
    class __SubclassedTestNode(__GenericTestNode):
        x: float
        z: float

    def teardown():
        directory._ast_node_types.pop(__GenericTestNode)
        directory._ast_node_types.pop(__SubclassedTestNode)

    request.addfinalizer(teardown)

    return __SubclassedTestNode


@pytest.mark.xfail(reason="Retrieving an Unregistered ASTNode", raises=directory.UnregisteredASTNode)
def test_unregistered_node_error():
    """Tests that Retrieiving an Unregistered Class Raises UnregisteredASTNode Exception."""
    directory.get_ast_node_type_info(int)


def test_register_ast_node(setup_ast_node):
    """Confirms we Correctly Register a New ASTNode Class"""

    assert issubclass(setup_ast_node, ASTNode), "Expected __GenericTestNode to be a Sublcass of ASTNode"
    assert setup_ast_node in directory._ast_node_types, "Expected __GenericTestNode to be Registered"

    ret = directory.get_ast_node_type_info(setup_ast_node)
    assert ret.fields == {"_span": Span, "x": int, "y": float}, "Expected Correct Annotations of __GenericTestNode"


def test_register_ast_node_with_bad_subclassing(setup_bad_ast_nodes):
    assert issubclass(setup_bad_ast_nodes, ASTNode), "Expected __SubclassedTestNode to be a Sublcass of ASTNode"
    assert setup_bad_ast_nodes in directory._ast_node_types, "Expected __SubclassedTestNode to be Registered"

    ret = directory.get_ast_node_type_info(setup_bad_ast_nodes)
    expected = {"_span": Span, "x": float, "y": float, "z": float}
    assert ret.fields == expected, "Expected Correct Annotation"
