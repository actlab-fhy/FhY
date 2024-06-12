"""Unit Test JSON Serialization Module, converting to and from AST Representation.

We use the intermediate data format, AlmostJson, to perform equality checks, because
our ir.type nodes within the AST are inherently unequal (__eq__ dunder methods
intentionally not implemented). We are also not concerned with the exact string order,
the data is serialized ('dumped') into as long as we are able to load that data, and is
equivalent to our starting data formats.

"""

import json
from typing import Callable, List, Tuple, TypeVar, Union

import pytest
from fhy.ir import (
    DataType,
    Identifier,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TupleType,
    TypeQualifier,
)
from fhy.lang.ast import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    BinaryOperation,
    ComplexLiteral,
    DeclarationStatement,
    Expression,
    ExpressionStatement,
    FloatLiteral,
    ForAllStatement,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
    ReturnStatement,
    SelectionStatement,
    TernaryExpression,
    TupleAccessExpression,
    TupleExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy.lang.ast.serialization.to_json import (
    AlmostJson,
    ASTtoJSON,
    JSONtoAST,
    dump,
    load,
)
from fhy.lang.ast.span import Source, Span

from ..utils import load_text

TLiteral = TypeVar("TLiteral", IntLiteral, FloatLiteral, ComplexLiteral)


def construct_id(name: str) -> Tuple[dict, Span]:
    """Build an Identifier from a Name Hint."""
    _id = Identifier(name)
    obj = dict(cls_name="Identifier", attributes=dict(name_hint=name, _id=_id._id))

    return obj, _id


@pytest.fixture
def span() -> Tuple[dict, Span]:
    a, b, c, d, e = 0, 10, 0, 10, "test"
    obj = dict(
        cls_name="Span",
        attributes=dict(
            start_line=a,
            end_line=b,
            start_column=c,
            end_column=d,
            source=dict(cls_name="Source", attributes=dict(namespace=e)),
        ),
    )
    sp = Span(a, b, c, d, Source(e))

    return obj, sp


@pytest.fixture
def literals(span) -> Callable[[Union[int, float, complex]], Tuple[dict, TLiteral]]:
    def _build(value: Union[int, float, complex]):
        span_obj, span_cls = span
        _literal: TLiteral
        if isinstance(value, complex):
            result = dict(real=value.real, imag=value.imag)
            _literal = ComplexLiteral(span=span_cls, value=value)
        elif isinstance(value, float):
            result = value
            _literal = FloatLiteral(span=span_cls, value=value)
        elif isinstance(value, int):
            result = value
            _literal = IntLiteral(span=span_cls, value=value)
        else:
            raise TypeError(f"Invalid Value Type: {value}")

        name = _literal.get_key_name()
        obj = dict(cls_name=name, attributes=dict(span=span_obj, value=result))
        return obj, _literal

    return _build


def build_numerical_type(
    primitive: PrimitiveDataType, shape_objs: List[dict], shape_cls: List[Expression]
) -> Tuple[dict, NumericalType]:
    """Builds a Numerical Type obj and node."""
    obj = dict(
        cls_name="NumericalType",
        attributes=dict(
            data_type=dict(
                cls_name="DataType",
                attributes=dict(primitive_data_type=primitive.value),
            ),
            shape=shape_objs,
        ),
    )

    numerical = NumericalType(
        data_type=DataType(primitive_data_type=primitive), shape=shape_cls
    )

    return obj, numerical


@pytest.fixture
def index_type(literals) -> Tuple[dict, IndexType]:
    one_obj, one_cls = literals(1)
    upper_obj, upper_cls = construct_id("k")

    obj = dict(
        cls_name="IndexType",
        attributes=dict(
            lower_bound=one_obj,
            upper_bound=upper_obj,
            stride=one_obj,
        ),
    )

    index = IndexType(
        lower_bound=one_cls,
        upper_bound=upper_cls,
        stride=one_cls,
    )

    return obj, index


@pytest.fixture
def tuple_type() -> Tuple[dict, TupleType]:
    shape_1_obj, shape_1_id = construct_id("m")
    num_obj, numerical = build_numerical_type(
        PrimitiveDataType.INT32, [shape_1_obj], [shape_1_id]
    )

    obj = dict(cls_name="TupleType", attributes=dict(types=[num_obj]))

    tup = TupleType(types=[numerical])

    return obj, tup


@pytest.fixture
def qualified(span) -> Tuple[dict, QualifiedType]:
    span_obj, span_cls = span
    shape_1_obj, shape_1_id = construct_id("m")
    shape_2_obj, shape_2_id = construct_id("n")
    num_obj, num_cls = build_numerical_type(
        PrimitiveDataType.INT32, [shape_1_obj, shape_2_obj], [shape_1_id, shape_2_id]
    )

    obj = dict(
        cls_name="QualifiedType",
        attributes=dict(
            span=span_obj,
            base_type=num_obj,
            type_qualifier="input",
        ),
    )

    qualified_type = QualifiedType(
        span=span_cls,
        base_type=num_cls,
        type_qualifier=TypeQualifier.INPUT,
    )

    return obj, qualified_type


@pytest.fixture
def arg1(span, qualified) -> Tuple[dict, Argument]:
    span_obj, span_cls = span
    qtype_obj, qtype_cls = qualified
    arg_id_obj, arg_id = construct_id("rupaul")

    obj = dict(
        cls_name="Argument",
        attributes=dict(
            span=span_obj,
            name=arg_id_obj,
            qualified_type=qtype_obj,
        ),
    )

    arg = Argument(
        span=span_cls,
        name=arg_id,
        qualified_type=qtype_cls,
    )

    return obj, arg


# EXPRESSIONS
@pytest.fixture
def unary(span, literals) -> Tuple[dict, UnaryExpression]:
    span_obj, span_cls = span
    negative = UnaryOperation.NEGATIVE
    literal_obj, literal_cls = literals(5)

    obj = dict(
        cls_name="UnaryExpression",
        attributes=dict(
            span=span_obj,
            operation=negative.value,
            expression=literal_obj,
        ),
    )

    _unary = UnaryExpression(span=span_cls, operation=negative, expression=literal_cls)

    return obj, _unary


@pytest.fixture
def binary(span, literals) -> Tuple[dict, BinaryExpression]:
    span_obj, span_cls = span
    addition = BinaryOperation.ADDITION
    lit_obj_left, lit_cls_left = literals(5)
    lit_obj_right, lit_cls_right = literals(10.0)

    obj = dict(
        cls_name="BinaryExpression",
        attributes=dict(
            span=span_obj,
            operation=addition.value,
            left=lit_obj_left,
            right=lit_obj_right,
        ),
    )

    bin_express = BinaryExpression(
        span=span_cls, operation=addition, left=lit_cls_left, right=lit_cls_right
    )

    return obj, bin_express


@pytest.fixture
def ternary(span, literals, binary, unary) -> Tuple[dict, TernaryExpression]:
    span_obj, span_cls = span
    cond_obj, cond_cls = literals(1)
    binary_obj, binary_cls = binary
    unary_obj, unary_cls = unary

    obj = dict(
        cls_name="TernaryExpression",
        attributes=dict(
            span=span_obj,
            condition=cond_obj,
            true=binary_obj,
            false=unary_obj,
        ),
    )

    expression = TernaryExpression(
        span=span_cls, condition=cond_cls, true=binary_cls, false=unary_cls
    )

    return obj, expression


@pytest.fixture
def tuple_access(span, literals) -> Tuple[dict, TupleAccessExpression]:
    span_obj, span_cls = span
    one_obj, one_cls = literals(1)
    name_obj, name_cls = construct_id("A")

    obj = dict(
        cls_name="TupleAccessExpression",
        attributes=dict(
            span=span_obj,
            tuple_expression=name_obj,
            element_index=one_obj,
        ),
    )
    access = TupleAccessExpression(
        span=span_cls,
        tuple_expression=name_cls,
        element_index=one_cls,
    )

    return obj, access


@pytest.fixture
def function_call(span) -> Tuple[dict, FunctionExpression]:
    span_obj, span_cls = span
    name_obj, name_cls = construct_id("funky")

    obj = dict(
        cls_name="FunctionExpression",
        attributes=dict(
            span=span_obj, function=name_obj, template_types=[], indices=[], args=[]
        ),
    )
    function = FunctionExpression(
        span=span_cls, function=name_cls, template_types=[], indices=[], args=[]
    )

    return obj, function


@pytest.fixture
def array_access(span) -> Tuple[dict, ArrayAccessExpression]:
    span_obj, span_cls = span
    array_id_obj, array_id_cls = construct_id("array")
    index_1_obj, index_1_cls = construct_id("j")
    index_2_obj, index_2_cls = construct_id("k")

    obj = dict(
        cls_name="ArrayAccessExpression",
        attributes=dict(
            span=span_obj,
            array_expression=array_id_obj,
            indices=[index_1_obj, index_2_obj],
        ),
    )
    array = ArrayAccessExpression(
        span=span_cls,
        array_expression=array_id_cls,
        indices=[index_1_cls, index_2_cls],
    )

    return obj, array


@pytest.fixture
def tuple_express(span) -> Tuple[dict, TupleExpression]:
    span_obj, span_cls = span

    obj = dict(
        cls_name="TupleExpression",
        attributes=dict(
            span=span_obj,
            expressions=[],
        ),
    )
    array = TupleExpression(
        span=span_cls,
        expressions=[],
    )

    return obj, array


@pytest.fixture
def id_express(span) -> Tuple[dict, IdentifierExpression]:
    span_obj, span_cls = span
    id_obj, id_cls = construct_id("name")

    obj = dict(
        cls_name="IdentifierExpression",
        attributes=dict(
            span=span_obj,
            identifier=id_obj,
        ),
    )
    expression = IdentifierExpression(
        span=span_cls,
        identifier=id_cls,
    )

    return obj, expression


# STATEMENTS
@pytest.fixture
def declaration(span, qualified, binary) -> Tuple[dict, DeclarationStatement]:
    span_obj, span_cls = span
    qtype_obj, qtype_cls = qualified
    binary_obj, binary_cls = binary
    varname_obj, varname_id = construct_id("bar")

    obj = dict(
        cls_name="DeclarationStatement",
        attributes=dict(
            span=span_obj,
            variable_name=varname_obj,
            variable_type=qtype_obj,
            expression=binary_obj,
        ),
    )

    statement = DeclarationStatement(
        span=span_cls,
        variable_name=varname_id,
        variable_type=qtype_cls,
        expression=binary_cls,
    )

    return obj, statement


@pytest.fixture
def express_state(span, unary, array_access) -> Tuple[dict, ExpressionStatement]:
    span_obj, span_cls = span
    array_obj, array_cls = array_access
    unary_obj, unary_cls = unary

    obj = dict(
        cls_name="ExpressionStatement",
        attributes=dict(
            span=span_obj,
            left=array_obj,
            right=unary_obj,
        ),
    )

    statement = ExpressionStatement(
        span=span_cls,
        left=array_cls,
        right=unary_cls,
    )

    return obj, statement


@pytest.fixture
def iteration_state(span) -> Tuple[dict, ForAllStatement]:
    span_obj, span_cls = span
    index_obj, index_cls = construct_id("elements")

    obj = dict(
        cls_name="ForAllStatement",
        attributes=dict(span=span_obj, index=index_obj, body=[]),
    )

    statement = ForAllStatement(span=span_cls, index=index_cls, body=[])

    return obj, statement


@pytest.fixture
def select_state(span, binary) -> Tuple[dict, SelectionStatement]:
    span_obj, span_cls = span
    binary_obj, binary_cls = binary

    obj = dict(
        cls_name="SelectionStatement",
        attributes=dict(
            span=span_obj, condition=binary_obj, true_body=[], false_body=[]
        ),
    )

    statement = SelectionStatement(
        span=span_cls, condition=binary_cls, true_body=[], false_body=[]
    )

    return obj, statement


@pytest.fixture
def return_state(span, unary) -> Tuple[dict, ReturnStatement]:
    span_obj, span_cls = span
    unary_obj, unary_cls = unary

    obj = dict(
        cls_name="ReturnStatement",
        attributes=dict(span=span_obj, expression=unary_obj),
    )

    statement = ReturnStatement(
        span=span_cls,
        expression=unary_cls,
    )

    return obj, statement


# FUNCTIONS
@pytest.fixture
def operation(span, arg1, qualified, express_state) -> Tuple[dict, Operation]:
    span_obj, span_cls = span
    arg1_obj, arg1_cls = arg1
    qtype_obj, qtype_cls = qualified
    name_id_obj, name_id = construct_id("foobar")
    e_state_obj, e_state_cls = express_state

    obj = dict(
        cls_name="Operation",
        attributes=dict(
            span=span_obj,
            name=name_id_obj,
            templates=[],
            args=[arg1_obj],
            body=[e_state_obj],
            return_type=qtype_obj,
        ),
    )

    op = Operation(
        span=span_cls,
        name=name_id,
        args=[arg1_cls],
        body=[e_state_cls],
        return_type=qtype_cls,
    )

    return obj, op


@pytest.fixture
def procedure(span, arg1, declaration) -> Tuple[dict, Procedure]:
    span_obj, span_cls = span
    arg1_obj, arg1_cls = arg1
    declare_obj, declare_cls = declaration
    name_id_obj, name_id = construct_id("buzz")

    obj = dict(
        cls_name="Procedure",
        attributes=dict(
            span=span_obj,
            name=name_id_obj,
            templates=[],
            args=[arg1_obj],
            body=[declare_obj],
        ),
    )

    proc = Procedure(
        span=span_cls, name=name_id, templates=[], args=[arg1_cls], body=[declare_cls]
    )

    return obj, proc


# MODULE
@pytest.fixture
def module(span, operation, procedure) -> Tuple[dict, Module]:
    span_obj, span_cls = span
    op_obj, op_cls = operation
    proc_obj, proc_cls = procedure

    obj = dict(
        cls_name="Module", attributes=dict(span=span_obj, statements=[op_obj, proc_obj])
    )

    module = Module(span=span_cls, statements=[op_cls, proc_cls])

    return obj, module


# FINALLY WE HAVE UNIT TESTS!
fixture_names = [
    "span",
    "index_type",
    "tuple_type",
    "qualified",
    "arg1",
    "unary",
    "binary",
    "ternary",
    "tuple_access",
    "function_call",
    "array_access",
    "tuple_express",
    "id_express",
    "declaration",
    "express_state",
    "iteration_state",
    "select_state",
    "return_state",
    "operation",
    "procedure",
]


def _test_to_json(obj, node):
    name = node.__class__.__name__
    test_obj = ASTtoJSON().visit(node).data()
    assert test_obj == obj, f"Resulting {name} Objects should be identical."


def _test_from_json(obj, node):
    node_text = dump(node)
    loaded = load_text(node_text)

    obj_text = json.dumps(obj)
    expected = load_text(obj_text)

    assert loaded == expected, "Expected same AlmostJson Objects after loading."

    ast_node = JSONtoAST().visit(loaded)
    assert isinstance(ast_node, node.__class__), "Expected same Node Type."

    assert isinstance(
        load(node_text), node.__class__
    ), "Expected same Node Type from load interface."

    assert isinstance(
        load(obj_text), node.__class__
    ), "Expected same Node Type from object using load interface."


@pytest.mark.parametrize("node_name", fixture_names)
@pytest.mark.parametrize("function", [_test_to_json, _test_from_json])
def test_ast_node_to_json(node_name, function, request):
    """Test Construction and Loading of a Node to AlmostJson Object."""
    fixture = request.getfixturevalue(node_name)
    obj, node = fixture
    function(obj, node)


@pytest.mark.parametrize("function", [_test_to_json, _test_from_json])
def test_shapeless_numerical_type(function):
    """Test Construction and Loading of Shapeless Numerical Type Node Json Object."""
    obj, numerical = build_numerical_type(PrimitiveDataType.FLOAT32, [], [])
    function(obj, numerical)


@pytest.mark.parametrize("function", [_test_to_json, _test_from_json])
def test_identifier_node(function):
    """Test Construction and Loading from Identifier Node to Json Object."""
    obj, node = construct_id("magic")
    function(obj, node)


@pytest.mark.parametrize("value", [int(1), float(1), complex(1)])
@pytest.mark.parametrize("function", [_test_to_json, _test_from_json])
def test_literal_values(value, function, literals):
    """Test Literal Value Node to and From Json Object conversions."""
    obj, node = literals(value)
    function(obj, node)


def test_module_to_json_object(module) -> None:
    """Confirm we construct the Same Object from a given Module AST Node."""
    obj, node = module
    result: AlmostJson = ASTtoJSON().visit_Module(node)

    assert (
        result.data() == obj
    ), "Resulting Json Object was not Constructed as expected."


@pytest.mark.skip(reason="We don't Particularly Care about the Order of Serialization.")
def test_module_json_str(module):
    """Test Conversion of Module AST Node to Json Intermediate Object."""
    obj, node = module
    indent = "  "
    # NOTE: This is harder to test, because the string order matters
    #       I'm not entirely convinced we care about this, as long as the objects
    #       (see test above) are the same.
    result: str = dump(node, indent)
    expected: str = json.dumps(obj, indent=indent)

    assert result == expected, "Resulting Json String was not serialized as expected."


def test_json_load(module):
    """Test Loading of Module Json String to AlmostJson Class."""
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
    _obj, node = module
    indent = "  "

    serialized: str = dump(node, indent)
    result = load(serialized)
    assert isinstance(result, Module), "Expected to Load an ast.Module Node."
    assert node.span == result.span, "Expected Identical Module Spans"

    # NOTE: This will not work because we cannot have type class equality
    #       That is, our node.statements != result.statements
    # assert node == result, "Expected Identical Module Nodes."

    assert (
        len(node.statements) == len(result.statements) == 2
    ), "Expected Two Statements"
    assert isinstance(result.statements[0], Operation), "Expected Operation Statement"
    assert (
        node.statements[0].name == result.statements[0].name
    ), "Expected Equivalent IDs"

    assert (
        len(node.statements[0].args) == len(result.statements[0].args) == 1
    ), "Expected 1 Argument"

    assert isinstance(result.statements[1], Procedure), "Expected Procedure Statement"


# FROM SOURCE CODE (Limited)
def test_empty_module(construct_ast):
    """Now create an empty module from Source code, and test Json conversion.

    Test that an Empty Module Serializes to Json object Correctly.

    """
    source = ""
    ast = construct_ast(source)

    expected = dict(
        cls_name="Module",
        attributes=dict(
            statements=[],
        ),
    )

    result: AlmostJson = ASTtoJSON().visit(ast)
    data = result.data()
    print(data)

    assert data == expected, "Unexpected Object."
