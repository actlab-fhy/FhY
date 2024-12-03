"""Tests the visitor pattern for the FhY AST."""

from collections.abc import Callable
from typing import TypeVar
from unittest.mock import MagicMock, patch

import pytest
from fhy.lang.ast import (
    ComplexLiteral,
    ExpressionStatement,
    IntLiteral,
    TupleAccessExpression,
    TupleExpression,
    visitor,
)
from fhy.lang.ast.alias import ASTStructure
from fhy.lang.ast.visitor import BasePass, Visitor
from fhy_core import CoreDataType, NumericalType, PrimitiveDataType

_BP = TypeVar("_BP", bound=BasePass)


@patch.object(Visitor, "default")
def test_unsupported_node_reaches_default(mock_method: MagicMock):
    """Test that unsupported node dispatches to default method."""
    visitor = Visitor()
    unsupported_node = MagicMock(name="UnsupportedNode")
    visitor.visit(unsupported_node)
    mock_method.assert_called_once_with(unsupported_node)


def test_visitor_unimplemented_visit_method():
    """Test behavior when a visit method for a specific node type is not implemented."""
    visitor = Visitor()
    unknown_node = MagicMock(name="Unknown")
    unknown_node.span = None

    with pytest.raises(NotImplementedError):
        visitor.visit(unknown_node)


def test_visitor_default_method():
    """Test the default method for unsupported nodes."""
    visitor = Visitor()
    unsupported_node = MagicMock(name="UnsupportedNode")
    with pytest.raises(NotImplementedError):
        visitor.default(unsupported_node)


def test_visitor_transform_expression_statement():
    """Test the transformer class transforms an expression statement node correctly."""
    transformer = visitor.Transformer()
    expression_statement_node = ExpressionStatement(
        span=None, left=None, right=IntLiteral(span=None, value=1)
    )
    result = transformer.visit_expression_statement(expression_statement_node)
    assert isinstance(result, ExpressionStatement)
    assert isinstance(result.right, IntLiteral)
    assert result.right.value == 1
    assert id(expression_statement_node) != id(result)
    assert id(expression_statement_node.right) != id(result.right)


def test_visitor_transform_tuple_expression():
    """Test the transformer class transforms a tuple expression node correctly."""
    transformer = visitor.Transformer()
    tuple_expression_node = TupleExpression(
        span=None, expressions=[IntLiteral(span=None, value=1)]
    )
    result = transformer.visit_tuple_expression(tuple_expression_node)
    assert isinstance(result, TupleExpression)
    assert isinstance(result.expressions[0], IntLiteral)
    assert result.expressions[0].value == 1

    assert id(tuple_expression_node) != id(result)
    assert id(tuple_expression_node.expressions[0]) != id(result.expressions[0])


def test_visitor_transform_tuple_access_expression():
    """Test the transformer class transforms a tuple access expression
    node correctly.
    """
    transformer = visitor.Transformer()
    tuple_access_expression_node = TupleAccessExpression(
        span=None,
        tuple_expression=IntLiteral(span=None, value=1),
        element_index=IntLiteral(span=None, value=0),
    )
    result = transformer.visit_tuple_access_expression(tuple_access_expression_node)
    assert isinstance(result, TupleAccessExpression)
    assert result.element_index.value == 0

    assert id(tuple_access_expression_node) != id(result), "Expected Shallow copy."
    assert id(tuple_access_expression_node.tuple_expression) != id(
        result.tuple_expression
    ), "Expected Shallow copy."
    assert id(tuple_access_expression_node.element_index) != id(
        result.element_index
    ), "Expected Shallow copy."


def test_visitor_transform_complex_literal():
    """Test the transformer class transforms a complex literal node correctly."""
    transformer = visitor.Transformer()
    value = complex(1025, 4097)
    complex_literal_node = ComplexLiteral(span=None, value=value)
    result = transformer.visit_complex_literal(complex_literal_node)
    assert isinstance(result, ComplexLiteral)
    assert result.value == value

    assert id(complex_literal_node) != id(result)
    assert id(complex_literal_node.value) == id(result.value)


def test_visitor_transform_type():
    """Test the transformer class transforms a type node correctly."""
    transformer = visitor.Transformer()
    primitive_data_type = CoreDataType.FLOAT32
    numerical_type_node = NumericalType(
        data_type=PrimitiveDataType(core_data_type=primitive_data_type),
        shape=[],
    )
    result = transformer.visit_type(numerical_type_node)
    assert isinstance(result, NumericalType)


def mock_transform(x):
    return x


def default_assert(
    instance: _BP, method: MagicMock, node: ASTStructure, transform=mock_transform
):
    # We check `__call__`, generic `visit`, and mocked `method` directly.
    for call in (instance, instance.visit, method):
        call(node)
        method.assert_called_once_with(transform(node))
        method.reset_mock()


def visit_generic(
    name: str, obj: _BP, _test=default_assert, transform=mock_transform
) -> Callable[[ASTStructure], _BP]:
    """Dynamic Construction of a mock patched base pass."""

    @patch.object(obj, name)
    def inner(node, mock_method) -> _BP:
        instance = obj()
        method = getattr(instance, name)

        assert isinstance(method, MagicMock), f"Method {name} is not properly mocked."
        assert isinstance(
            mock_method, MagicMock
        ), f"Method {name} is not properly mocked."
        assert method == mock_method, "Should be same mock method..."
        _test(instance, method, node, transform)

        return instance

    return inner


fixtures = [
    # Types
    ("index_type", "visit_index_type"),
    ("tuple_type", "visit_tuple_type"),
    ("qualified", "visit_qualified_type"),
    ("arg1", "visit_argument"),
    # Expressions
    ("unary", "visit_unary_expression"),
    ("binary", "visit_binary_expression"),
    ("ternary", "visit_ternary_expression"),
    ("tuple_access", "visit_tuple_access_expression"),
    ("function_call", "visit_function_expression"),
    ("array_access", "visit_array_access_expression"),
    ("tuple_express", "visit_tuple_expression"),
    ("id_express", "visit_identifier_expression"),
    # Statements
    ("declaration", "visit_declaration_statement"),
    ("express_state", "visit_expression_statement"),
    ("iteration_state", "visit_for_all_statement"),
    ("select_state", "visit_selection_statement"),
    ("return_state", "visit_return_statement"),
    ("import_node", "visit_import"),
    # Functions
    ("operation", "visit_operation"),
    ("procedure", "visit_procedure"),
    ("module", "visit_module"),
]


@pytest.mark.parametrize(["cls_name"], [(Visitor,), (visitor.Transformer,)])
@pytest.mark.parametrize(["fixture_name", "method_name"], fixtures)
def test_visit_ast_node_fixtures(
    fixture_name: str, method_name: str, cls_name: _BP, request: pytest.FixtureRequest
):
    """Test core functionality of the visitor base pass pattern: dispatching.

    Use statically constructed ast fixtures coupled to the method name we expect
    to be called for a given node. This doesn't validate downstream behavior of
    the given visit method in itself. That is, we only confirm we correctly visit the
    root node.

    """
    _obj, node = request.getfixturevalue(fixture_name)
    mocker: Callable[[ASTStructure], _BP] = visit_generic(
        method_name, cls_name, default_assert
    )
    instance: _BP = mocker(node)


def test_module_node(module):
    """Test correct nodes are visited from a Module node."""
    obj, node = module

    def mock_assert(instance: _BP, method: MagicMock, _node: ASTStructure, t):
        instance.visit(_node)
        method.assert_called_once_with(t(_node))

    for s, transform in (
        ("visit_sequence", lambda x: x.statements),
        ("visit_operation", lambda x: x.statements[0]),
        ("visit_procedure", lambda x: x.statements[1]),
    ):
        mocker: Callable[[ASTStructure], _BP] = visit_generic(
            s, Visitor, mock_assert, transform
        )
        mocker(node)
