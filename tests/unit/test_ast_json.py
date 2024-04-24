"""

"""
from typing import Tuple

import pytest

from fhy.ir import (
    DataType,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
)
from fhy.lang.ast import Argument, Module, Native, Operation, Procedure, QualifiedType
from fhy.lang.printer.to_json import ASTtoJSON, dump
from fhy.lang.span import Slice, Source, Span


@pytest.fixture(scope="module")
def span() -> Tuple[dict, Span]:
    obj = {
        "cls_name": "Span",
        "attributes": {
            "start_line": 0,
            "end_line": 10,
            "start_column": 0,
                "end_column": 20,
                "source": {
                    "cls_name": "Source",
                    "attributes": {
                        "namespace": "test"
                    }
                }
        }
    }
    sp = Span(0, 10, 0, 20, Source("test"))
    return obj, sp


def construct_id(name: str) -> Tuple[dict, Span]:
    _id = Identifier(name)
    obj = {
        "cls_name": "Identifier",
        "attributes": {
            "name_hint": name,
            "_id": _id._id,
        }
    }

    return obj, _id


def qualified_type():
    ...

@pytest.fixture(scope="module")
def arg1(span) -> Tuple[dict, Argument]:
    span_obj, span_cls = span
    arg_id_obj, arg_id = construct_id("rupaul")
    shape_1_obj, shape_1_id = construct_id("m")
    shape_2_obj, shape_2_id = construct_id("n")

    obj = {
        "cls_name": "Argument",
        "attributes": {
            "span": span_obj,
            "name": arg_id_obj,
            "qualified_type": {
                "cls_name": "QualifiedType",
                "attributes": {
                    "span": span_obj,
                    "base_type": {
                        "cls_name": "NumericalType",
                        "attributes": {
                            "data_type": {
                                "cls_name": "DataType",
                                "attributes": {
                                    "primitive_data_type": "int32"
                                }
                            },
                            "shape": [shape_1_obj, shape_2_obj],
                        }
                    },
                    "type_qualifier": "input"
                }
            },
        }
    }

    arg = Argument(
        span=span_cls,
        name=arg_id,
        qualified_type=QualifiedType(
            span=span_cls,
            base_type=NumericalType(
                data_type=DataType(
                    primitive_data_type=PrimitiveDataType.INT32
                ),
                shape=[shape_1_id, shape_2_id]
            ),
            type_qualifier=TypeQualifier.INPUT
        )
    )

    return obj, arg


@pytest.fixture(scope="module")
def operation(span, arg1) -> Tuple[dict, Operation]:
    span_obj, span_cls = span
    arg1_obj, arg1_cls = arg1
    name_id_obj, name_id = construct_id("bar")

    obj = {
        "cls_name": "Operation",
        "attributes": {
            "span": span_obj,
            "name": name_id_obj,
            "args": [arg1_obj],
            "body": [],
            "return_type": {
                "cls_name": "QualifiedType",
                "attributes": {
                    "span": span_obj,
                    "base_type": {
                        "cls_name": "NumericalType",
                        "attributes": {
                            "data_type": {
                                "cls_name": "DataType",
                                "attributes": {
                                    "primitive_data_type": "int32"
                                }
                            },
                            "shape": [],
                        }
                    },
                    "type_qualifier": "output"
                }
            }
        }
    }

    op = Operation(
        span=span_cls,
        name=name_id,
        args=[arg1_cls],
        body=[],
        return_type=QualifiedType(
            span=span_cls,
            base_type=NumericalType(
                data_type=DataType(
                    primitive_data_type=PrimitiveDataType.INT32
                ),
                shape=[]
            ),
            type_qualifier=TypeQualifier.OUTPUT,
        )
    )

    return obj, op


@pytest.fixture(scope="module")
def module(span, operation) -> Tuple[dict, Module]:
    span_obj, span_cls = span
    op_obj, op_cls = operation

    obj = {
        "cls_name": "Module",
        "attributes": {
            "span": span_obj,
            "components": [op_obj]
        }
    }

    module = Module(
        span=span_cls,
        components=[op_cls]
    )

    return obj, module


def test_module_to_json_object(module):
    obj, node = module

    result: dict = ASTtoJSON().visit_Module(node)

    assert result == obj, "Resulting Json Object was not Constructed as expected."
