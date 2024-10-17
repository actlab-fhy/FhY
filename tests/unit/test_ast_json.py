"""Unit Test JSON Serialization Module, converting to and from AST Representation.

We use the intermediate data format, AlmostJson, to perform equality checks, because
our ir.type nodes within the AST are inherently unequal (__eq__ dunder methods
intentionally not implemented). We are also not concerned with the exact string order,
the data is serialized ('dumped') into as long as we are able to load that data, and is
equivalent to our starting data formats.

"""

import json
from typing import TypeVar

import pytest
from fhy.ir import (
    CoreDataType,
)
from fhy.lang.ast import (
    ComplexLiteral,
    FloatLiteral,
    IntLiteral,
    Module,
    Operation,
    Procedure,
)
from fhy.lang.ast.serialization.to_json import (
    AlmostJson,
    ASTtoJSON,
    JSONtoAST,
    dump,
    load,
)

from ..utils import load_text
from .conftest import fixture_node_names as fixture_names

TLiteral = TypeVar("TLiteral", IntLiteral, FloatLiteral, ComplexLiteral)


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
def test_shapeless_numerical_type(function, build_numerical_type):
    """Test Construction and Loading of Shapeless Numerical Type Node Json Object."""
    obj, numerical = build_numerical_type(CoreDataType.FLOAT32, [], [])
    function(obj, numerical)


@pytest.mark.parametrize("function", [_test_to_json, _test_from_json])
def test_identifier_node(function, construct_id):
    """Test Construction and Loading from Identifier Node to Json Object."""
    obj, node = construct_id("magic")
    function(obj, node)


@pytest.mark.parametrize("value", [1, float(1), complex(1)])
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


@pytest.mark.parametrize(
    ["method"], [(i,) for i in dir(JSONtoAST) if i.startswith("visit_")]
)
def test_to_ast_none_nodes(method):
    """Confirm visit methods raise value error when node supplied is None."""
    if method in ("visit_Span", "visit_sequence", "visit_Source"):
        ...
    else:
        serial = JSONtoAST()
        function = getattr(serial, method)
        with pytest.raises(ValueError):
            function(None)


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

    assert data == expected, "Unexpected Object."
