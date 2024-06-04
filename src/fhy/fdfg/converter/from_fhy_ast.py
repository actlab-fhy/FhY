from typing import Any, Optional, Callable
import networkx as nx
from fhy.lang import ast
from fhy.ir import Identifier, TypeQualifier, Type
from fhy.fdfg.core import Node, FractalizedNode, PrimitiveNode, Edge, SourceNode, SinkNode, FDFG
import fhy.fdfg.ops as fdfg_op


class FhYASTExpressionConverter(ast.BasePass):
    """Creates an unfinished f-DFG from an AST expression.

    In this instance, unfinished means that the f-DFG edges are not assigned to
    intermediate variables and, therefore, do not have associated types yet.

    On a call, the converter will create a MultiDiGraph representing the f-DFG
    with source nodes for each symbol in the expression and a sink node for the
    result of the expression with a mocked symbol named "output".
    """
    _UNARY_OPERATION_TO_FDFG_OP = {
        ast.UnaryOperation.NEGATIVE: fdfg_op.neg_op,
        ast.UnaryOperation.BITWISE_NOT: fdfg_op.bitwise_not_op,
        ast.UnaryOperation.LOGICAL_NOT: fdfg_op.logical_not_op,
    }
    _BINARY_OPERATION_TO_FDFG_OP = {
        ast.BinaryOperation.ADDITION: fdfg_op.add_op,
        ast.BinaryOperation.SUBTRACTION: fdfg_op.sub_op,
        ast.BinaryOperation.MULTIPLICATION: fdfg_op.mul_op,
        ast.BinaryOperation.DIVISION: fdfg_op.div_op,
        ast.BinaryOperation.FLOORDIV: fdfg_op.floor_div_op,
        ast.BinaryOperation.MODULO: fdfg_op.mod_op,
        ast.BinaryOperation.POWER: fdfg_op.pow_op,
        ast.BinaryOperation.BITWISE_AND: fdfg_op.bitwise_and_op,
        ast.BinaryOperation.BITWISE_OR: fdfg_op.bitwise_or_op,
        ast.BinaryOperation.BITWISE_XOR: fdfg_op.bitwise_xor_op,
        ast.BinaryOperation.LEFT_SHIT: fdfg_op.left_shift_op,
        ast.BinaryOperation.RIGHT_SHIFT: fdfg_op.right_shift_op,
        ast.BinaryOperation.LESS_THAN: fdfg_op.less_than_op,
        ast.BinaryOperation.LESS_THAN_OR_EQUAL: fdfg_op.less_than_or_equal_op,
        ast.BinaryOperation.GREATER_THAN: fdfg_op.greater_than_op,
        ast.BinaryOperation.GREATHER_THAN_OR_EQUAL: fdfg_op.greater_than_or_equal_op,
        ast.BinaryOperation.EQUAL_TO: fdfg_op.equal_op,
        ast.BinaryOperation.NOT_EQUAL_TO: fdfg_op.not_equal_op,
    }

    _graph: nx.MultiDiGraph

    def __init__(self) -> None:
        super().__init__()
        self._graph = nx.MultiDiGraph()

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    def __call__(self, node: ast.Expression) -> None:
        if not isinstance(node, ast.Expression):
            raise RuntimeError(f"{self.__class__.__name__} only accepts \
                               AST expressions, but got {type(node)}.")

        final_expression_node_name = self.visit(node)

        sink_node_name = Identifier("output")
        sink_node = SinkNode(Identifier("output"))
        self._add_node_to_graph(sink_node_name, sink_node)
        self._add_edge_to_graph(final_expression_node_name, sink_node_name)

    def visit(self, node: ast.Expression) -> Identifier:
        return super().visit(node)

    def visit_UnaryExpression(self, node: ast.UnaryExpression) -> Identifier:
        expression_node_name = self.visit(node.expression)

        op = self._UNARY_OPERATION_TO_FDFG_OP[node.operation]

        primitive_node_name = Identifier(op.name.name_hint)
        primitive_node = PrimitiveNode(op)
        self._add_node_to_graph(primitive_node_name, primitive_node)
        self._add_edge_to_graph(expression_node_name, primitive_node_name)
        return primitive_node_name

    def visit_BinaryExpression(
        self,
        node: ast.BinaryExpression
    ) -> Identifier:
        left_expression_node_name = self.visit(node.left)
        right_expression_node_name = self.visit(node.right)

        op = self._BINARY_OPERATION_TO_FDFG_OP[node.operation]

        primitive_node_name = Identifier(op.name.name_hint)
        primitive_node = PrimitiveNode(op)
        self._add_node_to_graph(primitive_node_name, primitive_node)
        self._add_edge_to_graph(left_expression_node_name, primitive_node_name)
        self._add_edge_to_graph(right_expression_node_name, primitive_node_name)
        return primitive_node_name

    def visit_IdentifierExpression(
        self,
        node: ast.IdentifierExpression
    ) -> Identifier:
        source_node_name = Identifier(node.identifier.name_hint)
        source_node = SourceNode(node.identifier)
        self._add_node_to_graph(source_node_name, source_node)
        return source_node_name

    def _add_edge_to_graph(
        self,
        source_node_name: Identifier,
        sink_node_name: Identifier
    ) -> None:
        self._graph.add_edge(source_node_name, sink_node_name)

    def _add_node_to_graph(self, node_name: Identifier, node: Node) -> None:
        self._graph.add_node(node_name, data=node)


def _convert_fhy_ast_expression_to_graph(
    node: ast.Expression
) -> nx.MultiDiGraph:
    converter = FhYASTExpressionConverter()
    converter(node)
    return converter.graph


# class FhYASTFunctionConverter(ast.BasePass):
#     _function_graph: nx.MultiDiGraph
#     _symbol_to_source_node: dict[Identifier, Node]

#     def __init__(self) -> None:
#         super().__init__()
#         self._function_graph = nx.MultiDiGraph()
#         self._symbol_to_source_node = {}

#     def visit_Procedure(self, node: ast.Procedure) -> None:
#         node_name = Identifier(node.name.name_hint)

#         # For each argument, create a source node if the argument is an input, param, or state
#         # and create a sink node if the argument is an output or state
#         # Add the node to the multi di-graph
#         input_node_names = []
#         output_node_names = []
#         for arg in node.args:
#             arg_node_name = Identifier(arg.name.name_hint)
#             arg_type_qualifier = arg.qualified_type.type_qualifier
#             if arg_type_qualifier in (TypeQualifier.INPUT, TypeQualifier.PARAM, TypeQualifier.STATE):
#                 source_node = SourceNode(arg_node_name, arg.name)
#                 self._add_node_to_graph(source_node)
#                 self._update_symbol_to_source_node_mapping(arg.name, source_node)
#                 input_node_names.append(arg_node_name)
#             if arg_type_qualifier in (TypeQualifier.OUTPUT, TypeQualifier.STATE):
#                 sink_node = SinkNode(arg_node_name, arg.name)
#                 self._add_node_to_graph(sink_node)
#                 self._update_symbol_to_source_node_mapping(arg.name, sink_node)
#                 output_node_names.append(arg_node_name)

#         self._visit_list(node.body)

#     def _visit_list(self, nodes: list[Any]) -> None:
#         for node in nodes:
#             self.visit(node)

#     def visit_ExpressionStatement(self, node: ast.ExpressionStatement) -> None:
#         if isinstance(node.left, ast.IdentifierExpression):
#             symbol_name = node.left.identifier
#             self._update_symbol_to_source_node_mapping(symbol_name, final_expression_node)
#         else:
#             raise NotImplementedError(f"FhYASTFunctionConverter does not \
#                                       support assignment to \
#                                       \"{type(node.left)}\" expressions.")

#     def _get_source_node_for_symbol(self, symbol_name: Identifier) -> Node:
#         return self._symbol_to_source_node[symbol_name]

#     def _update_symbol_to_source_node_mapping(self, symbol_name: Identifier, source_node: Node) -> None:
#         self._symbol_to_source_node[symbol_name] = source_node

#     def _add_edge_to_graph(self, source_node: Node, sink_node: Node) -> None:
#         self._function_graph.add_edge(source_node, sink_node)

#     def _add_node_to_graph(self, node: Node) -> None:
#         self._function_graph.add_node(node)


# def from_fhy_ast_function(node: ast.Function) -> FractalizedNode:
#     converter = FhYASTFunctionConverter()
#     converter(node)
#     graph = FDFG()
#     graph._graph = converter._function_graph
#     return FractalizedNode(Identifier(node.name.name_hint), [], [], graph)
