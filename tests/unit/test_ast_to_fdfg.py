import pytest

from typing import Any
import networkx as nx
from fhy.fdfg.converter.from_fhy_ast import (
    _convert_fhy_ast_expression_to_graph,
    # from_fhy_ast_function
)
from fhy.fdfg.visualize import plot_fdfg
from fhy.lang import ast
from fhy.ir import Identifier
from fhy.lang.ast.pprint import pformat_ast
from fhy.fdfg.core import FDFG, SourceNode, SinkNode, PrimitiveNode, FractalizedNode
import fhy.fdfg.ops as fdfg_op


def assert_expected_expression_graph(
    graph: Any,
    expected_graph: nx.MultiDiGraph
) -> None:
    assert isinstance(graph, nx.MultiDiGraph), f"Expected a MultiDiGraph \
        object, but got {type(graph)}."

    def node_match(n1, n2):
        n1_data = n1["data"]
        n2_data = n2["data"]
        if isinstance(n1_data, SourceNode) and isinstance(n2_data, SourceNode):
            return n1_data.symbol_name == n2_data.symbol_name
        elif isinstance(n1_data, FractalizedNode) and isinstance(n2_data, FractalizedNode):
            raise NotImplementedError("FractalizedNode is not implemented.")
        elif isinstance(n1_data, PrimitiveNode) and isinstance(n2_data, PrimitiveNode):
            return n1_data.op == n2_data.op
        elif isinstance(n1_data, SinkNode) and isinstance(n2_data, SinkNode):
            return True
        else:
            raise RuntimeError(f"Unexpected node type: {type(n1_data)} and {type(n2_data)}.")

    assert nx.algorithms.is_isomorphic(graph, expected_graph,
                                       node_match=node_match), f"Graphs are not isomorphic. Expected: {expected_graph}, but got: {graph}."


@pytest.mark.parametrize(
    ["ast_unary_operation", "expected_fdfg_op"],
    [
        (ast.UnaryOperation.NEGATIVE, fdfg_op.neg_op),
        (ast.UnaryOperation.BITWISE_NOT, fdfg_op.bitwise_not_op),
        (ast.UnaryOperation.LOGICAL_NOT, fdfg_op.logical_not_op),
    ]
)
def test_unary_neg_graph(ast_unary_operation, expected_fdfg_op):
    a = Identifier("a")
    expression_ast = ast.UnaryExpression(
        operation=ast_unary_operation,
        expression=ast.IdentifierExpression(identifier=a)
    )

    a_node = SourceNode(a)
    neg_node = PrimitiveNode(expected_fdfg_op)
    output_node = SinkNode(Identifier(""))
    expected_graph = nx.MultiDiGraph()
    expected_graph.add_node("a", data=a_node)
    expected_graph.add_node("neg", data=neg_node)
    expected_graph.add_node("output", data=output_node)
    expected_graph.add_edge("a", "neg")
    expected_graph.add_edge("neg", "output")

    graph = _convert_fhy_ast_expression_to_graph(expression_ast)

    assert_expected_expression_graph(graph, expected_graph)


@pytest.mark.parametrize(
    ["ast_binary_operation", "expected_fdfg_op"],
    [
        (ast.BinaryOperation.ADDITION, fdfg_op.add_op),
        (ast.BinaryOperation.SUBTRACTION, fdfg_op.sub_op),
        (ast.BinaryOperation.MULTIPLICATION, fdfg_op.mul_op),
        (ast.BinaryOperation.DIVISION, fdfg_op.div_op),
        (ast.BinaryOperation.FLOORDIV, fdfg_op.floor_div_op),
        (ast.BinaryOperation.MODULO, fdfg_op.mod_op),
        (ast.BinaryOperation.POWER, fdfg_op.pow_op),
        (ast.BinaryOperation.BITWISE_AND, fdfg_op.bitwise_and_op),
        (ast.BinaryOperation.BITWISE_OR, fdfg_op.bitwise_or_op),
        (ast.BinaryOperation.BITWISE_XOR, fdfg_op.bitwise_xor_op),
        (ast.BinaryOperation.LEFT_SHIT, fdfg_op.left_shift_op),
        (ast.BinaryOperation.RIGHT_SHIFT, fdfg_op.right_shift_op),
        (ast.BinaryOperation.LESS_THAN, fdfg_op.less_than_op),
        (ast.BinaryOperation.LESS_THAN_OR_EQUAL, fdfg_op.less_than_or_equal_op),
        (ast.BinaryOperation.GREATER_THAN, fdfg_op.greater_than_op),
        (ast.BinaryOperation.GREATHER_THAN_OR_EQUAL, fdfg_op.greater_than_or_equal_op),
        (ast.BinaryOperation.EQUAL_TO, fdfg_op.equal_op),
        (ast.BinaryOperation.NOT_EQUAL_TO, fdfg_op.not_equal_op),
    ]
)
def test_binary_operation_graph(ast_binary_operation, expected_fdfg_op):
    a = Identifier("a")
    b = Identifier("b")
    expression_ast = ast.BinaryExpression(
        operation=ast_binary_operation,
        left=ast.IdentifierExpression(identifier=a),
        right=ast.IdentifierExpression(identifier=b)
    )

    a_node = SourceNode(a)
    b_node = SourceNode(b)
    add_node = PrimitiveNode(expected_fdfg_op)
    output_node = SinkNode(Identifier(""))
    expected_graph = nx.MultiDiGraph()
    expected_graph.add_node("a", data=a_node)
    expected_graph.add_node("b", data=b_node)
    expected_graph.add_node("add", data=add_node)
    expected_graph.add_node("output", data=output_node)
    expected_graph.add_edge("a", "add")
    expected_graph.add_edge("b", "add")
    expected_graph.add_edge("add", "output")

    graph = _convert_fhy_ast_expression_to_graph(expression_ast)

    assert_expected_expression_graph(graph, expected_graph)
