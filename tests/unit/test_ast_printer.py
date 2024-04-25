from fhy.ir import (
    DataType,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
)
from fhy.lang.ast import Module, Operation, QualifiedType
from fhy.lang.printer import pprint_ast


def test_empty_program():
    ast = Module()

    output: str = pprint_ast(ast)

    assert output == ""


def test_empty_operation():
    operation_name = Identifier("foo")
    ast = Module(
        components=[
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

    output: str = pprint_ast(ast, is_identifier_id_printed=True)

    assert output == f"op (foo::{operation_name._id})() -> output int32 " + "{\n\n}"
