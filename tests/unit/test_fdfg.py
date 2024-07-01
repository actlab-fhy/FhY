import pytest

from typing import Any
import networkx as nx
from fhy.ir import Identifier
from fhy.fdfg.core import FDFG, SourceNode, SinkNode
from fhy.fdfg.node.fractalized import FractalizedNode, FunctionNode
from fhy.fdfg.node.parametric import LoopNode, ReductionNode
from fhy.fdfg.edge import Edge
from fhy.fdfg.node.primitive import PrimitiveNode
import fhy.fdfg.ops as fdfg_op


def create_default_expected_fdfg() -> tuple[Identifier, Identifier, nx.MultiDiGraph]:
    source_node_name = Identifier("source")
    sink_node_name = Identifier("sink")
    expected_graph = nx.MultiDiGraph()
    expected_graph.add_node(source_node_name, data=SourceNode())
    expected_graph.add_node(sink_node_name, data=SinkNode())
    return source_node_name, sink_node_name, expected_graph


def assert_expected_expression_graph(
    graph: Any,
    expected_graph: nx.MultiDiGraph,
) -> None:
    assert isinstance(graph, nx.MultiDiGraph), f"Expected a MultiDiGraph \
        object, but got {type(graph)}."

    def node_match(n1, n2):
        n1_data = n1["data"]
        n2_data = n2["data"]
        if isinstance(n1_data, SourceNode) and isinstance(n2_data, SourceNode):
            return True
        elif isinstance(n1_data, FunctionNode) and isinstance(n2_data, FunctionNode):
            try:
                assert_expected_expression_graph(n1_data.fdfg.graph, n2_data.fdfg.graph)
            except AssertionError as e:
                return False
            return True
        elif isinstance(n1_data, PrimitiveNode) and isinstance(n2_data, PrimitiveNode):
            return n1_data.op == n2_data.op
        elif isinstance(n1_data, SinkNode) and isinstance(n2_data, SinkNode):
            return True
        else:
            raise RuntimeError(f"Unexpected node type: {type(n1_data)} and {type(n2_data)}.")

    def edge_match(e1, e2):
        if len(e1) != len(e2):
            return False
        for e1_attr, e2_attr in zip(e1.values(), e2.values()):
            e1_data = e1_attr["data"]
            e2_data = e2_attr["data"]
            # TODO: Add commutative primitive operations support
            if e1_data != e2_data:
                return False
        return True

    assert nx.algorithms.is_isomorphic(graph, expected_graph,
                                       node_match=node_match,
                                       edge_match=edge_match), f"Graphs are not isomorphic. Expected: {expected_graph}, but got: {graph}."


def test_empty_fdfg():
    fdfg = FDFG()

    assert fdfg.graph.number_of_nodes() == 2, f"Expected 2 nodes, got {fdfg.graph.number_of_nodes()}."
    assert fdfg.graph.number_of_edges() == 0, f"Expected 0 edges, got {fdfg.graph.number_of_edges()}."
    assert fdfg.source_node_name in fdfg.graph.nodes(), f"Expected source node in graph nodes."
    assert fdfg.sink_node_name in fdfg.graph.nodes(), f"Expected sink node in graph nodes."
    assert isinstance(fdfg.graph.nodes()[fdfg.source_node_name]["data"], SourceNode), f"Expected source node to be a SourceNode, got {type(fdfg.graph.nodes()[fdfg.source_node_name]['data'])}."
    assert isinstance(fdfg.graph.nodes()[fdfg.sink_node_name]["data"], SinkNode), f"Expected sink node to be a SinkNode, got {type(fdfg.graph.nodes()[fdfg.sink_node_name]['data'])}."


def test_get_number_of_nodes():
    fdfg = FDFG()

    assert fdfg.get_number_of_nodes() == fdfg.graph.number_of_nodes(), f"Expected number of nodes to be {fdfg.graph.number_of_nodes()}, got {fdfg.get_number_of_nodes()}."


# TODO: add more basic unit tests for FDFG class


@pytest.mark.parametrize(
    ["given_fdfg_op"],
    [
        (fdfg_op.neg_op,),
        (fdfg_op.bitwise_not_op,),
        (fdfg_op.logical_not_op,),
    ]
)
def test_basic_primitive_unary_operation_graph(given_fdfg_op):
    fdfg = FDFG()
    a = Identifier("a")
    c = Identifier("c")
    unary_op_node_name = Identifier("unary_op")
    unary_op_node = PrimitiveNode(given_fdfg_op)
    expected_graph_source_node_name, expected_graph_sink_node_name, expected_graph = create_default_expected_fdfg()
    expected_graph_unary_op_node_name = Identifier("op")
    expected_graph.add_node(expected_graph_unary_op_node_name, data=PrimitiveNode(given_fdfg_op))
    expected_graph.add_edge(expected_graph_source_node_name, expected_graph_unary_op_node_name, data=Edge(a, 0, 0))
    expected_graph.add_edge(expected_graph_unary_op_node_name, expected_graph_sink_node_name, data=Edge(c, 0, 0))

    fdfg.add_node(unary_op_node_name, unary_op_node)
    fdfg.add_edge(fdfg.source_node_name, unary_op_node_name, 0, 0, symbol_name=a)
    fdfg.add_edge(unary_op_node_name, fdfg.sink_node_name, 0, 0, symbol_name=c)

    assert_expected_expression_graph(fdfg.graph, expected_graph)


@pytest.mark.parametrize(
    ["given_fdfg_op"],
    [
        (fdfg_op.add_op,),
        (fdfg_op.sub_op,),
        (fdfg_op.mul_op,),
        (fdfg_op.div_op,),
        (fdfg_op.floor_div_op,),
        (fdfg_op.mod_op,),
        (fdfg_op.pow_op,),
        (fdfg_op.bitwise_and_op,),
        (fdfg_op.bitwise_or_op,),
        (fdfg_op.bitwise_xor_op,),
        (fdfg_op.left_shift_op,),
        (fdfg_op.right_shift_op,),
        (fdfg_op.less_than_op,),
        (fdfg_op.less_than_or_equal_op,),
        (fdfg_op.greater_than_op,),
        (fdfg_op.greater_than_or_equal_op,),
        (fdfg_op.equal_op,),
        (fdfg_op.not_equal_op,),
    ]
)
def test_basic_primitive_binary_operation_graph(given_fdfg_op):
    fdfg = FDFG()
    a = Identifier("a")
    b = Identifier("b")
    c = Identifier("c")
    binary_op_node_name = Identifier("binary_op")
    binary_op_node = PrimitiveNode(given_fdfg_op)
    expected_graph_source_node_name, expected_graph_sink_node_name, expected_graph = create_default_expected_fdfg()
    expected_graph_binary_op_node_name = Identifier("op")
    expected_graph.add_node(expected_graph_binary_op_node_name, data=PrimitiveNode(given_fdfg_op))
    expected_graph.add_edge(expected_graph_source_node_name, expected_graph_binary_op_node_name, data=Edge(a, 0, 0))
    expected_graph.add_edge(expected_graph_source_node_name, expected_graph_binary_op_node_name, data=Edge(b, 1, 1))
    expected_graph.add_edge(expected_graph_binary_op_node_name, expected_graph_sink_node_name, data=Edge(c, 0, 0))

    fdfg.add_node(binary_op_node_name, binary_op_node)
    fdfg.add_edge(fdfg.source_node_name, binary_op_node_name, 0, 0, symbol_name=a)
    fdfg.add_edge(fdfg.source_node_name, binary_op_node_name, 1, 1, symbol_name=b)
    fdfg.add_edge(binary_op_node_name, fdfg.sink_node_name, 0, 0, symbol_name=c)

    assert_expected_expression_graph(fdfg.graph, expected_graph)


def test_basic_function_call_graph():
    function_fdfg = FDFG()
    a = Identifier("a")
    b = Identifier("b")
    c = Identifier("c")
    op_node_name = Identifier("op")
    op_node = PrimitiveNode(fdfg_op.add_op)
    function_fdfg.add_node(op_node_name, op_node)
    function_fdfg.add_edge(function_fdfg.source_node_name, op_node_name, 0, 0, symbol_name=a)
    function_fdfg.add_edge(function_fdfg.source_node_name, op_node_name, 1, 1, symbol_name=b)
    function_fdfg.add_edge(op_node_name, function_fdfg.sink_node_name, 0, 0, symbol_name=c)
    fdfg = FDFG()
    x = Identifier("x")
    y = Identifier("y")
    z = Identifier("z")
    function_node_name = Identifier("function_node")
    function_name = Identifier("function")
    function_node = FunctionNode(function_name, fdfg=function_fdfg)
    fdfg.add_node(function_node_name, function_node)
    fdfg.add_edge(fdfg.source_node_name, function_node_name, 0, 0, symbol_name=x)
    fdfg.add_edge(fdfg.source_node_name, function_node_name, 1, 1, symbol_name=y)
    fdfg.add_edge(function_node_name, fdfg.sink_node_name, 0, 0, symbol_name=z)
    expected_function_graph_source_node_name, expected_function_graph_sink_node_name, expected_function_graph = create_default_expected_fdfg()
    expected_function_graph_op_node_name = Identifier("op")
    expected_function_graph.add_node(expected_function_graph_op_node_name, data=PrimitiveNode(fdfg_op.add_op))
    expected_function_graph.add_edge(expected_function_graph_source_node_name, expected_function_graph_op_node_name, data=Edge(a, 0, 0))
    expected_function_graph.add_edge(expected_function_graph_source_node_name, expected_function_graph_op_node_name, data=Edge(b, 1, 1))
    expected_function_graph.add_edge(expected_function_graph_op_node_name, expected_function_graph_sink_node_name, data=Edge(c, 0, 0))
    expected_graph_source_node_name, expected_graph_sink_node_name, expected_graph = create_default_expected_fdfg()
    expected_graph_function_node_name = Identifier("function")
    expected_graph.add_node(expected_graph_function_node_name, data=FunctionNode(function_name, fdfg=FDFG(expected_function_graph)))
    expected_graph.add_edge(expected_graph_source_node_name, expected_graph_function_node_name, data=Edge(x, 0, 0))
    expected_graph.add_edge(expected_graph_source_node_name, expected_graph_function_node_name, data=Edge(y, 1, 1))
    expected_graph.add_edge(expected_graph_function_node_name, expected_graph_sink_node_name, data=Edge(z, 0, 0))

    assert_expected_expression_graph(fdfg.graph, expected_graph)
