""" """

from typing import Callable, Generator, Optional, Type, TypeVar

import pytest

from fhy import ir
from fhy.lang.ast import IntLiteral, directory
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

T = TypeVar("T")


@pytest.fixture
def register_node() -> Generator[Callable[[Type[T]], Type[T]], None, None]:
    """Registers an AST Node and Removes Node from Registry on Teardown"""
    node_cls: Optional[Type[T]] = None

    def _inner(node_type: Type[T]) -> Type[T]:
        nonlocal node_cls
        node_cls = directory.register_ast_node(node_type)
        return node_cls

    yield _inner

    # Teardown
    if node_cls is not None:
        directory._ast_node_types.pop(node_cls)
        assert (
            node_cls not in directory._ast_node_types
        ), "Fixture Teardown Incomplete. Node Still Registered"


def test_builder_frame_attributes(register_node):
    """Tests that a Builder Frame populates Relevant Attributes of a given Node Type."""
    node_type: Type[Module] = register_node(Module)
    builder = ASTNodeBuilderFrame(node_type)

    # Attributes of Node should Exist in Builder
    for j in ("span", "components"):
        assert hasattr(builder, j), f"Expected Attribute `{j}`"

    assert not hasattr(
        builder, "_bad_attribute"
    ), "Builder should not have `_bad_attribute`"


def test_builder_frame_assignment_error(register_node):
    """Tests that Attempts to Assign an Irrelevant Attributes raises an Error."""
    node_type: Type[Module] = register_node(Module)
    builder = ASTNodeBuilderFrame(node_type)

    # Confirm users cannot Assign Attributes not Defined by the Node Class
    with pytest.raises(FieldAttributeError):
        builder._bad_apple = "Bad Apples"
    assert not hasattr(builder, "_bad_apple"), "User should not be able to Assign Attr."


def test_builder_frame_initial_attribute_values(register_node):
    """Tests that a Builder Frame assigns Default Values defined by Node Type."""
    node_type: Type[Module] = register_node(Module)
    builder = ASTNodeBuilderFrame(node_type)

    assert isinstance(builder.components, list), "Expected components to be a list."
    assert isinstance(builder.span, Span), "Expected span to be a Span"


def test_builder_frame_update(register_node):
    """Tests that a Builder Frame correctly Updates Attribute Values."""
    node_type: Type[Module] = register_node(Module)
    builder = ASTNodeBuilderFrame(node_type)

    builder.update(components=["test"])
    assert builder.components == ["test"], "Expected components to be updated"


def test_create_builder_frame(register_node):
    """Tests the Primary Entry Point function, create_builder_frame, works as
    expected.

    """
    node_type: Type[Module] = register_node(Module)
    result = create_builder_frame(node_type)

    assert isinstance(result, ASTNodeBuilderFrame), "Expected ASTNodeBuilderFrame"
    assert result.cls == Module, "Expected Module Class Defined by Builder."


@pytest.mark.parametrize(
    "node_cls, expected",
    [(ir.NumericalType, _NumericalTypeInfo), (ir.IndexType, _IndexTypeInfo)],
)
def test_create_type_builder_frame(register_node, node_cls, expected):
    """Tests the TypeBuilderFrame works as expected from API entry point"""
    node_type = register_node(node_cls)
    result = create_builder_frame(node_type)
    assert isinstance(result, TypeBuilderFrame), "Expected TypeBuilderFrame"
    assert isinstance(result._type_info, expected), f"Expected: {expected}"


def test_index_type_info_build(register_node):
    """Tests update and build of Index Type Node."""
    node_cls = ir.IndexType
    node_type = register_node(node_cls)
    builder: TypeBuilderFrame = create_builder_frame(node_type)
    lower = IntLiteral(value=1)
    upper = IntLiteral(value=5)
    builder.update(
        lower_bound=lower,
        upper_bound=upper,
    )
    # Builder correctly updated values
    assert builder._type_info.lower_bound == lower, "Expected Same Lower Bound"
    assert builder._type_info.upper_bound == upper, "Expected Same Upper Bound"

    # Builder correctly constructs node
    result = builder.build()
    assert isinstance(result, node_cls), "Expected Builder to construct IndexType"
