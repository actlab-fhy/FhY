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
from fhy.lang.ast import (
    Argument,
    BinaryExpression,
    BinaryOperation,
    DeclarationStatement,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
)
from fhy.lang.printer.to_json import AlmostJson, ASTtoJSON, dump, load, to_almost_json
from fhy.lang.span import Source, Span


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


@pytest.fixture(scope="module")
def int_literal(span) -> Tuple[dict, IntLiteral]:
    def _build(value: int):
        span_obj, span_cls = span
        obj = dict(
            cls_name="IntLiteral",
            attributes=dict(
                span=span_obj,
                value=value
            )
        )
        literal = IntLiteral(span=span_cls, value=value)
        return obj, literal

    return _build


@pytest.fixture(scope="module")
def qualified(span) -> Tuple[dict, QualifiedType]:
    span_obj, span_cls = span
    shape_1_obj, shape_1_id = construct_id("m")
    shape_2_obj, shape_2_id = construct_id("n")
    obj = {
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
    }

    qualified_type = QualifiedType(
        span=span_cls,
        base_type=NumericalType(
            data_type=DataType(primitive_data_type=PrimitiveDataType.INT32),
            shape=[shape_1_id, shape_2_id],
        ),
        type_qualifier=TypeQualifier.INPUT,
    )

    return obj, qualified_type


@pytest.fixture(scope="module")
def arg1(span, qualified) -> Tuple[dict, Argument]:
    span_obj, span_cls = span
    qtype_obj, qtype_cls = qualified
    arg_id_obj, arg_id = construct_id("rupaul")

    obj = {
        "cls_name": "Argument",
        "attributes": {
            "span": span_obj,
            "name": arg_id_obj,
            "qualified_type": qtype_obj,
        },
    }

    arg = Argument(
        span=span_cls,
        name=arg_id,
        qualified_type=qtype_cls,
    )

    return obj, arg


@pytest.fixture(scope="module")
def binary(span, int_literal) -> Tuple[dict, BinaryExpression]:
    span_obj, span_cls = span
    addition = BinaryOperation.ADDITION
    lit_obj_left, lit_cls_left = int_literal(5)
    lit_obj_right, lit_cls_right = int_literal(10)

    obj = {
        "cls_name": "BinaryExpression",
        "attributes": {
            "span": span_obj,
            "operation": addition.value,
            "left": lit_obj_left,
            "right": lit_obj_right,
        }
    }

    bexpress = BinaryExpression(
        span=span_cls,
        operation=addition,
        left=lit_cls_left,
        right=lit_cls_right
    )

    return obj, bexpress


@pytest.fixture(scope="module")
def declaration(span, qualified, binary) -> Tuple[dict, Operation]:
    span_obj, span_cls = span
    qtype_obj, qtype_cls = qualified
    binary_obj, binary_cls = binary
    varname_obj, varname_id = construct_id("bar")

    obj = {
        "cls_name": "DeclarationStatement",
        "attributes": {
            "span": span_obj,
            "variable_name": varname_obj,
            "variable_type": qtype_obj,
            "expression": binary_obj,
        }
    }

    statement = DeclarationStatement(
        span=span_cls,
        variable_name=varname_id,
        variable_type=qtype_cls,
        expression=binary_cls
    )

    return obj, statement


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
def procedure(span, arg1, declaration) -> Tuple[dict, Procedure]:
    span_obj, span_cls = span
    arg1_obj, arg1_cls = arg1
    declare_obj, declare_cls = declaration
    name_id_obj, name_id = construct_id("bar")

    obj = {
        "cls_name": "Procedure",
        "attributes": {
            "span": span_obj,
            "name": name_id_obj,
            "args": [arg1_obj],
            "body": [declare_obj],
        },
    }

    op = Procedure(
        span=span_cls,
        name=name_id,
        args=[arg1_cls],
        body=[declare_cls]
    )

    return obj, op


@pytest.fixture(scope="module")
def module(span, operation, procedure) -> Tuple[dict, Module]:
    span_obj, span_cls = span
    op_obj, op_cls = operation
    proc_obj, proc_cls = procedure

    obj = {
        "cls_name": "Module",
        "attributes": {"span": span_obj, "components": [op_obj, proc_obj]},
    }

    module = Module(span=span_cls, components=[op_cls, proc_cls])

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


def test_load_json_to_ast(module):
    """Test Serialization of AST Node and reloading that text returns same Object.
    In other words: ASTNode --> JSONtext --> ASTNode

    """
    obj, node = module
    indent = "  "

    serialized: str = dump(node, indent)
    result = load(serialized)
    assert isinstance(result, Module), "Expected to Load an ast.Module Node."
    assert node.span == result.span, "Expected Identical Module Spans"

    # NOTE: This will not work because we cannot have type class equality
    #       That is, our node.components != result.components
    # assert node == result, "Expected Identical Module Nodes."

    assert (
        len(node.components) == len(result.components) == 2
    ), "Expected single Component"
    assert isinstance(result.components[0], Operation), "Expected Operation Component"
    assert (
        node.components[0].name == result.components[0].name
    ), "Expected Equivalent IDs"

    assert (
        len(node.components[0].args) == len(result.components[0].args) == 1
    ), "Expected 1 Argument"
