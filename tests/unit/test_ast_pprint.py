"""Unit Test the Pretty Print Module."""

from fhy.lang.ast import Module, Operation, QualifiedType
from fhy.lang.ast.pprint import pformat_ast
from fhy_core import (
    CoreDataType,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
)


def test_empty_program():
    """Test Pretty Printing an Empty Program Module AST."""
    ast = Module()

    output: str = pformat_ast(ast)

    assert output == ""


def test_empty_operation():
    """Test Pretty Printing a Module AST with an Operation Body component."""
    operation_name = Identifier("foo")
    ast = Module(
        statements=[
            Operation(
                name=operation_name,
                args=[],
                return_type=QualifiedType(
                    type_qualifier=TypeQualifier.OUTPUT,
                    base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32), []),
                ),
                body=[],
            )
        ]
    )

    output: str = pformat_ast(ast, show_id=True)

    assert output == f"op (foo::{operation_name._id})<>() -> output int32 " + "{\n\n}"


# TYPES
def test_index_type(index_type):
    """Test pretty printing index type node."""
    obj, node = index_type
    expected = "index[1:k:1]"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_tuple_type(tuple_type):
    """Test pretty printing tuple type node."""
    obj, node = tuple_type
    expected = "tuple ( int32[m], )"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


# EXPRESSIONS
def test_unary_expression(unary):
    """Test pretty printing unary expression node."""
    obj, node = unary
    expected = "-(5)"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_binary_expression(binary):
    """Test pretty printing binary expression node."""
    obj, node = binary
    expected = "(5 + 10.0)"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_ternary_expression(ternary):
    """Test pretty printing ternary expression node."""
    obj, node = ternary
    expected = "(1 ? (5 + 10.0) : -(5))"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_tuple_access_expression(tuple_access):
    """Test pretty printing tuple access expression node."""
    obj, node = tuple_access
    expected = "A.1"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_function_expression(function_call):
    """Test pretty printing function expression node."""
    obj, node = function_call
    expected = "funky<>[]()"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_array_access_expression(array_access):
    """Test pretty printing array access expression node."""
    obj, node = array_access
    expected = "array[j, k]"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_tuple_expression(tuple_express):
    """Test pretty printing tuple expression node."""
    obj, node = tuple_express
    expected = "(  )"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_identifier_expression(id_express):
    """Test pretty printing identifier expression node."""
    obj, node = id_express
    expected = "name"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


# STATEMENTS
def test_declaration_statement(declaration):
    """Test pretty printing declaration statement node."""
    obj, node = declaration
    expected = "input int32[m, n] bar = (5 + 10.0);"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_expression_statement(express_state):
    """Test pretty printing expression statement node."""
    obj, node = express_state
    expected = "array[j, k] = -(5);"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_iteration_statement(iteration_state):
    """Test pretty printing iteration statement node."""
    obj, node = iteration_state
    expected = "forall (elements) {\n\n}"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_selection_statement(select_state):
    """Test pretty printing selection statement node."""
    obj, node = select_state
    expected = "if (5 + 10.0) {\n\n}"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_return_statement(return_state):
    """Test pretty printing return statement node."""
    obj, node = return_state
    expected = "return -(5);"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


# FUNCTIONS
def test_operation(operation):
    """Test pretty printing operation node."""
    obj, node = operation
    expected = "op foobar<>(input int32[m, n] rupaul) -> input int32[m, n]"
    expected += " {\n  array[j, k] = -(5);\n}"
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_procedure(procedure):
    """Test pretty printing operation node."""
    obj, node = procedure
    expected: str = (
        "proc buzz<>(input int32[m, n] rupaul) "
        "{\n  input int32[m, n] bar = (5 + 10.0);\n}"
    )
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"


def test_procedure_with_templates(procedure_with_templates):
    obj, node = procedure_with_templates
    expected: str = "proc mumu<T>() {\n\n}"

    result = pformat_ast(node)
    assert result == expected, "Unexpected Formatting"


def test_module(module):
    """Test pretty printing module node."""
    obj, node = module
    expected: str = (
        "op foobar<>(input int32[m, n] rupaul) -> input int32[m, n]"
        " {\n  array[j, k] = -(5);\n}\n"
        "proc buzz<>(input int32[m, n] rupaul) "
        "{\n  input int32[m, n] bar = (5 + 10.0);\n}"
    )
    result = pformat_ast(node)

    assert result == expected, "Unexpected Formatting"
