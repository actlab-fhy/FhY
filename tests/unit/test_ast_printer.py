import pytest

from fhy.lang.ast import Argument, Module, Native, Operation, Procedure, QualifiedType
from fhy.ir import DataType, Identifier, Type, TypeQualifier, NumericalType, PrimitiveDataType
from fhy.lang.printer import pprint_ast


def test_empty_program():
    ast = Module()

    output: str = pprint_ast(ast)

    assert output == ""


def test_empty_operation():
    Identifier._next_id = 0
    operation_name = Identifier("foo")
    ast = Module(
        components=[
            Operation(
                name=Identifier("foo"),
                args=[],
                return_type=QualifiedType(type_qualifier=TypeQualifier.OUTPUT, base_type=NumericalType(DataType(PrimitiveDataType.INT32), [])),
                body=[]
            )
        ]
    )

    output: str = pprint_ast(ast)

    assert output == "op foo::0() -> output int32 {}"
