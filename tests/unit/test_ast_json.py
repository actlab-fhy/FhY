""" """

import json
from typing import Tuple

import pytest

from fhy.ir import (
    DataType,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
)
from fhy.lang.ast import Argument, Module, Operation, QualifiedType
from fhy.lang.printer.to_json import AlmostJson, ASTtoJSON, dump, load, to_almost_json
from fhy.lang.span import Source, Span


@pytest.fixture(scope="module")
def span() -> Tuple[dict, Span]:
    obj = {
        "cls_name": "Span",
        "attributes": {
            "start_line": 0,
            "end_line": 10,
            "start_column": 0,
            "end_column": 20,
            "source": {"cls_name": "Source", "attributes": {"namespace": "test"}},
        },
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
        },
    }

    return obj, _id


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
                                "attributes": {"primitive_data_type": "int32"},
                            },
                            "shape": [shape_1_obj, shape_2_obj],
                        },
                    },
                    "type_qualifier": "input",
                },
            },
        },
    }

    arg = Argument(
        span=span_cls,
        name=arg_id,
        qualified_type=QualifiedType(
            span=span_cls,
            base_type=NumericalType(
                data_type=DataType(primitive_data_type=PrimitiveDataType.INT32),
                shape=[shape_1_id, shape_2_id],
            ),
            type_qualifier=TypeQualifier.INPUT,
        ),
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
                                "attributes": {"primitive_data_type": "int32"},
                            },
                            "shape": [],
                        },
                    },
                    "type_qualifier": "output",
                },
            },
        },
    }

    op = Operation(
        span=span_cls,
        name=name_id,
        args=[arg1_cls],
        body=[],
        return_type=QualifiedType(
            span=span_cls,
            base_type=NumericalType(
                data_type=DataType(primitive_data_type=PrimitiveDataType.INT32),
                shape=[],
            ),
            type_qualifier=TypeQualifier.OUTPUT,
        ),
    )

    return obj, op


@pytest.fixture(scope="module")
def module(span, operation) -> Tuple[dict, Module]:
    span_obj, span_cls = span
    op_obj, op_cls = operation

    obj = {
        "cls_name": "Module",
        "attributes": {"span": span_obj, "components": [op_obj]},
    }

    module = Module(span=span_cls, components=[op_cls])

    return obj, module


def test_module_to_json_object(module) -> None:
    obj, node = module

    result: dict = ASTtoJSON().visit_Module(node)

    assert result == obj, "Resulting Json Object was not Constructed as expected."


# def test_module_json_str(module):
#     obj, node = module
#     indent = "  "
#     # NOTE: This is harder to test, because the string order matters
#     #       I'm not entirely convinced we care about this, as long as the objects
#     #       (see test above) are the same.
#     result: str = dump(node, indent)
#     expected: str = json.dumps(obj, indent=indent)

#     assert result == expected, "Resulting Json String was not serialized as expected."


def load_text(text: str):
    return json.loads(text, object_hook=to_almost_json)


def test_json_load(module):
    """Tests Loading of Json String to AlmostJson Class"""
    obj, node = module
    indent = "  "
    serialized: str = dump(node, indent)
    result = load_text(serialized)
    assert isinstance(result, AlmostJson), "Expected loaded object to be AlmostJSON cls"

    expected_str: str = json.dumps(obj)
    expected_obj = load_text(expected_str)

    assert result == expected_obj, "Expected AlmostJson Objects to be Equal"
