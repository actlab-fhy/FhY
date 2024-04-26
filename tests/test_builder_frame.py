""" """

import pytest

from fhy import ir
from fhy.lang.ast import directory
from fhy.lang.ast.core import Module
from fhy.lang.ast_builder.builder_frame import (
    ASTNodeBuilderFrame,
    FieldAttributeError,
    TypeBuilderFrame,
    _IndexTypeInfo,
    _NumericalTypeInfo,
    create_builder_frame,
)
from fhy.lang.span import Span


@pytest.fixture
def register_node(request):
    """Registers an AST Node and Removes Node from Registry on Teardown"""
    # Dynamic Fixture Argument (of another Fixture)
    node = request.param
    yield directory.register_ast_node(node)

    # Teardown
    directory._ast_node_types.pop(node)
    assert (
        node not in directory._ast_node_types
    ), "Fixture Teardown Incomplete. Node Still Registered"


@pytest.mark.parametrize("register_node", [Module], indirect=True)
def test_builder_frame_attributes(register_node):
    builder = ASTNodeBuilderFrame(register_node)

    # Attributes of Node should Exist in Builder
    for j in ("span", "components"):
        assert hasattr(builder, j), f"Expected Attribute `{j}`"

    assert not hasattr(
        builder, "_bad_attribute"
    ), "Builder should not have `_bad_attribute`"


@pytest.mark.parametrize("register_node", [Module], indirect=True)
def test_builder_frame_assignment_error(register_node):
    builder = ASTNodeBuilderFrame(register_node)

    # Confirm users cannot Assign Attributes not Defined by the Node Class
    with pytest.raises(FieldAttributeError):
        builder._bad_apple = "Bad Apples"
    assert not hasattr(builder, "_bad_apple"), "User should not be able to Assign Attr."


@pytest.mark.parametrize("register_node", [Module], indirect=True)
def test_builder_frame_initial_attribute_values(register_node):
    builder = ASTNodeBuilderFrame(register_node)

    # We expect Initial Values of attributes to take on default values from node cls
    assert isinstance(builder.components, list), "Expected components to be a list."
    assert isinstance(builder.span, Span), "Expected span to be a Span"


@pytest.mark.parametrize("register_node", [Module], indirect=True)
def test_builder_frame_update(register_node):
    builder = ASTNodeBuilderFrame(register_node)

    builder.update(components=["test"])
    assert builder.components == ["test"], "Expected components to be updated"


@pytest.mark.parametrize("register_node", [Module], indirect=True)
def test_create_builder_frame(register_node):
    result = create_builder_frame(register_node)
    assert isinstance(result, ASTNodeBuilderFrame), "Expected ASTNodeBuilderFrame"


@pytest.mark.parametrize("register_node, expected", [
                         (ir.NumericalType, _NumericalTypeInfo),
                         (ir.IndexType, _IndexTypeInfo)
                         ],
                         indirect=["register_node"])
def test_create_type_builder_frame(register_node, expected):
    result = create_builder_frame(register_node)
    assert isinstance(result, TypeBuilderFrame), "Expected TypeBuilderFrame"
    assert isinstance(result._type_info, expected), f"Expected: {expected}"
