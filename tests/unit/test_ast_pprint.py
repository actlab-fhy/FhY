"""Unit Test the Pretty Print Module."""

from fhy.ir import (
    DataType,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
)
from fhy.lang.ast import Module, Operation, QualifiedType
from fhy.lang.ast.pprint import pformat_ast


# TODO: Complete Coverage of AST Pretty Print Module in Unit Testing.
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
                    base_type=NumericalType(DataType(PrimitiveDataType.INT32), []),
                ),
                body=[],
            )
        ]
    )

    output: str = pformat_ast(ast, is_identifier_id_printed=True)

    assert output == f"op (foo::{operation_name._id})<>() -> output int32 " + "{\n\n}"
