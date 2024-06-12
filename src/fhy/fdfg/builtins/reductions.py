from fhy.ir.identifier import Identifier
import networkx as nx
from ..core import FDFG
from ..edge import Edge
from ..node.fractalized import FunctionNode, FractalizedNode
from ..node.primitive import PrimitiveNode
from ..node.parametric import LoopNode
from ..node.io import SourceNode, SinkNode
from ..op import Op
import fhy.fdfg.ops as fdfg_op
from typing import Any


BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS: dict[str, Identifier] = {
    "sum": Identifier("sum"),
    "prod": Identifier("prod"),
    "min": Identifier("min"),
    "max": Identifier("max"),
}


_BUILTIN_REDUCTION_FUNCTION_OPS: dict[str, Op] = {
    "sum": fdfg_op.add_op,
    "prod": fdfg_op.mul_op,
}


def generate_reduction_fdfg(
    reduction_function_name: str,
    reduction_indices: list[Identifier],
    expression_fdfg: FDFG,
) -> FDFG:
    raise NotImplementedError()
    # reduction_body_fdfg = FDFG()
    # expression_fdfg_node_name = Identifier("expression")
    # reduction_body_fdfg.add_node(
    #     expression_fdfg_node_name, FractalizedNode(expression_fdfg)
    # )
    # for node in expression_fdfg.get_input_nodes():
    #     source_node_name = Identifier(node.symbol_name.name_hint)
    #     reduction_body_fdfg.add_node(source_node_name, SourceNode(node.symbol_name))
    #     reduction_body_fdfg.add_edge(source_node_name, expression_fdfg_node_name)

    # carry_in_node_name = Identifier(f"{reduction_function_name}_carry_in")
    # carry_in_node = SourceNode(carry_in_node_name)
    # operation_node_name = Identifier(
    #     _BUILTIN_REDUCTION_FUNCTION_OPS[reduction_function_name].name
    # )
    # operation_node = PrimitiveNode(
    #     _BUILTIN_REDUCTION_FUNCTION_OPS[reduction_function_name]
    # )
    # output_node_name = Identifier("output")
    # output_node = SinkNode(output_node_name)
    # reduction_body_fdfg.add_node(carry_in_node_name, carry_in_node)
    # reduction_body_fdfg.add_node(operation_node_name, operation_node)
    # reduction_body_fdfg.add_node(output_node_name, output_node)
    # reduction_body_fdfg.add_edge(carry_in_node_name, operation_node_name)
    # reduction_body_fdfg.add_edge(expression_fdfg_node_name, operation_node_name)
    # reduction_body_fdfg.add_edge(operation_node_name, output_node_name)

    # loop_node_name = Identifier(
    #     f"{reduction_function_name}_loop_{'_'.join([index.name_hint for index in reduction_indices])}"
    # )
    # loop_node = LoopNode(set(reduction_indices), reduction_body_fdfg)

    # reduction_fdfg = FDFG()
    # reduction_fdfg.add_node(loop_node_name, loop_node)
    # for node in expression_fdfg.get_input_nodes():
    #     source_node_name = Identifier(node.symbol_name.name_hint)
    #     reduction_fdfg.add_node(source_node_name, SourceNode(node.symbol_name))
    #     reduction_fdfg.add_edge(source_node_name, loop_node_name)
    # output_node_name = Identifier("output")
    # output_node = SinkNode(output_node_name)
    # reduction_fdfg.add_node(output_node_name, output_node)
    # reduction_fdfg.add_edge(loop_node_name, output_node_name)
    # return reduction_fdfg
