from typing import Any, Optional, Callable, Union
from abc import ABC
import networkx as nx
from fhy.lang.ast.visitor import BasePass as ASTBasePass
import fhy.lang.ast.node as ast
from fhy.lang.ast.passes import collect_indices
from fhy.ir.identifier import Identifier
from fhy.ir.type import TypeQualifier, Type, IndexType
from fhy.ir.table import SymbolTable, VariableSymbolTableFrame
from fhy.fdfg.core import Node, SourceNode, SinkNode, FDFG
from fhy.fdfg.node.fractalized import FunctionNode, FractalizedNode
from fhy.fdfg.node.parametric import LoopNode
from fhy.fdfg.node.io import SinkNode, SourceNode
from fhy.fdfg.node.primitive import PrimitiveNode, LiteralNode
from fhy.fdfg.edge import Edge
import fhy.fdfg.ops as fdfg_op
from fhy.utils import Stack
from fhy.fdfg.builtins import (
    generate_reduction_fdfg,
    BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS,
)


class FhYASTExpressionConverter(ASTBasePass):
    """Creates an unfinished f-DFG from an AST expression.

    In this instance, unfinished means that the f-DFG edges are not assigned to
    intermediate variables and, therefore, do not have associated types yet.

    On a call, the converter will create a MultiDiGraph representing the f-DFG
    with source nodes for each symbol in the expression and a sink node for the
    result of the expression with a mocked symbol named "output".

    Note:
        A key assumption made by this converter is that ever expression only
        has one output.
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
    _symbol_to_source_node_name: dict[Identifier, Identifier]

    def __init__(self) -> None:
        super().__init__()
        self._graph = nx.MultiDiGraph()
        self._symbol_to_source_node_name = {}

    @property
    def fdfg(self) -> FDFG:
        fdfg = FDFG()
        fdfg._graph = self._graph
        return fdfg

    def __call__(self, node: ast.Expression) -> None:
        if not isinstance(node, ast.Expression):
            raise RuntimeError(
                f"{self.__class__.__name__} expects AST expressions, but got {type(node)}."
            )

        final_expression_node_name = self.visit(node)

        # Mock a sink node for the expression f-DFG
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

    def visit_BinaryExpression(self, node: ast.BinaryExpression) -> Identifier:
        left_expression_node_name = self.visit(node.left)
        right_expression_node_name = self.visit(node.right)

        op = self._BINARY_OPERATION_TO_FDFG_OP[node.operation]

        primitive_node_name = Identifier(op.name.name_hint)
        primitive_node = PrimitiveNode(op)
        self._add_node_to_graph(primitive_node_name, primitive_node)
        self._add_edge_to_graph(left_expression_node_name, primitive_node_name)
        self._add_edge_to_graph(right_expression_node_name, primitive_node_name)
        return primitive_node_name

    def visit_FunctionExpression(self, node: ast.FunctionExpression) -> Identifier:
        # TODO: support indices both given to the function and in the arguments
        if (
            isinstance(node.function, ast.IdentifierExpression)
            and node.function.identifier
            in BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS.values()
        ):
            return self._visit_reduction_function_expression(node)
        elif isinstance(node.function, ast.IdentifierExpression):
            arg_node_names = [self.visit(arg) for arg in node.args]
            if isinstance(node.function, ast.IdentifierExpression):
                function_name = node.function
                function_node_name = Identifier(function_name.identifier.name_hint)
                function_node = FunctionNode(symbol_name=node.function.identifier)
                self._add_node_to_graph(function_node_name, function_node)
                for arg_node_name in arg_node_names:
                    self._add_edge_to_graph(arg_node_name, function_node_name)
                return function_node_name
            else:
                raise NotImplementedError(
                    f'FhYASTExpressionConverter does not support functions defined as "{type(node.function)}" expressions.'
                )

    def _visit_reduction_function_expression(
        self, node: ast.FunctionExpression
    ) -> Identifier:
        if len(node.args) != 1:
            raise RuntimeError("Expected only one argument for a reduction function.")

        reduction_expression_fdfg = _convert_fhy_ast_expression_to_fdfg(node.args[0])

        reduction_function_name: Identifier = node.function.identifier

        # TODO: the indices field should be a list of identifiers rather than a list of expressions...
        node_indices = [index.identifier for index in node.indices]
        reduction_fdfg = generate_reduction_fdfg(
            reduction_function_name.name_hint, node_indices, reduction_expression_fdfg
        )

        reduction_fdfg_node_name = Identifier(reduction_function_name.name_hint)
        reduction_fdfg_node = FractalizedNode(reduction_fdfg)
        self._add_node_to_graph(reduction_fdfg_node_name, reduction_fdfg_node)
        return reduction_fdfg_node_name

    def visit_ArrayAccessExpression(
        self, node: ast.ArrayAccessExpression
    ) -> Identifier:
        return self.visit(node.array_expression)

    def visit_IdentifierExpression(self, node: ast.IdentifierExpression) -> Identifier:
        symbol_name = node.identifier
        if symbol_name not in self._symbol_to_source_node_name:
            source_node_name = Identifier(symbol_name.name_hint)
            source_node = SourceNode(symbol_name)
            self._add_node_to_graph(source_node_name, source_node)
            self._symbol_to_source_node_name[symbol_name] = source_node_name
        return self._symbol_to_source_node_name[symbol_name]

    def visit_IntLiteral(
        self,
        node: ast.IntLiteral,
    ) -> Identifier:
        literal_node_name = Identifier(str(node.value))
        literal_node = LiteralNode(node.value)
        self._add_node_to_graph(literal_node_name, literal_node)
        return literal_node_name

    def visit_FloatLiteral(
        self,
        node: ast.FloatLiteral,
    ) -> Identifier:
        literal_node_name = Identifier(str(node.value))
        literal_node = LiteralNode(node.value)
        self._add_node_to_graph(literal_node_name, literal_node)
        return literal_node_name

    def _merge_fdfg(self, fdfg: FDFG) -> nx.MultiDiGraph:
        self._function_graph = nx.compose(self._function_graph, fdfg._graph)

        for input_node_name, input_node in zip(
            fdfg.get_input_node_names(), fdfg.get_input_nodes()
        ):
            adjacent_nodes = list(fdfg.graph.successors(input_node_name))
            for adjacent_node in adjacent_nodes:
                self._add_edge_to_graph(
                    self._symbol_to_source_node_name[input_node.symbol_name],
                    adjacent_node,
                )
            self._remove_node_from_graph(input_node_name)

    def _add_edge_to_graph(
        self, source_node_name: Identifier, sink_node_name: Identifier
    ) -> None:
        self._graph.add_edge(source_node_name, sink_node_name)

    def _add_node_to_graph(self, node_name: Identifier, node: Node) -> None:
        self._graph.add_node(node_name, data=node)

    def _remove_node_from_graph(self, node_name: Identifier) -> None:
        self._graph.remove_node(node_name)


def _convert_fhy_ast_expression_to_fdfg(node: ast.Expression) -> FDFG:
    converter = FhYASTExpressionConverter()
    converter(node)
    return converter.fdfg


class FhYASTFunctionConverter(ASTBasePass):
    _function_graph: nx.MultiDiGraph
    _symbol_to_source_node_name: dict[Identifier, Identifier]

    _symbol_table: SymbolTable
    _function_name: Optional[Identifier]

    def __init__(self, symbol_table: SymbolTable) -> None:
        super().__init__()
        self._function_graph = nx.MultiDiGraph()
        self._symbol_to_source_node_name = {}

        self._symbol_table = symbol_table
        self._function_name = None

    def __call__(self, node: ast.Function) -> Any:
        if not isinstance(node, (ast.Function)):
            raise RuntimeError(
                f"{self.__class__.__name__} expects AST functions, but got {type(node)}."
            )
        self._function_name = node.name
        return super().__call__(node)

    def visit_Procedure(self, node: ast.Procedure) -> None:
        node_name = Identifier(node.name.name_hint)

        input_node_names = []
        for arg in node.args:
            arg_type_qualifier = arg.qualified_type.type_qualifier
            if arg_type_qualifier in (
                TypeQualifier.INPUT,
                TypeQualifier.PARAM,
                TypeQualifier.STATE,
            ):
                arg_node_name = Identifier(arg.name.name_hint)
                source_node = SourceNode(arg.name)
                self._add_node_to_graph(arg_node_name, source_node)
                self._update_symbol_to_source_node_mapping(arg.name, arg_node_name)
                input_node_names.append(arg_node_name)

        self._visit_list(node.body)

        output_node_names = []
        for arg in node.args:
            arg_type_qualifier = arg.qualified_type.type_qualifier
            if arg_type_qualifier in (TypeQualifier.OUTPUT, TypeQualifier.STATE):
                arg_node_name = Identifier(arg.name.name_hint)
                sink_node = SinkNode(arg.name)
                self._add_node_to_graph(arg_node_name, sink_node)
                self._add_edge_to_graph(
                    self._get_source_node_name_for_symbol(arg.name), arg_node_name
                )
                # self._update_symbol_to_source_node_mapping(arg.name, arg_node_name)
                output_node_names.append(arg_node_name)

    def visit_Operation(self, node: ast.Operation) -> None:
        node_name = Identifier(node.name.name_hint)

        input_node_names = []
        for arg in node.args:
            arg_type_qualifier = arg.qualified_type.type_qualifier
            if arg_type_qualifier in (
                TypeQualifier.INPUT,
                TypeQualifier.PARAM,
                TypeQualifier.STATE,
            ):
                arg_node_name = Identifier(arg.name.name_hint)
                source_node = SourceNode(arg.name)
                self._add_node_to_graph(arg_node_name, source_node)
                self._update_symbol_to_source_node_mapping(arg.name, arg_node_name)
                input_node_names.append(arg_node_name)

        self._visit_list(node.body)

    def _visit_list(self, nodes: list[Any]) -> None:
        for node in nodes:
            self.visit(node)

    def visit_DeclarationStatement(self, node: ast.DeclarationStatement) -> None:
        if isinstance(node.variable_type.base_type, IndexType):
            return

        id_node_name = Identifier(node.variable_name.name_hint)
        id_node = PrimitiveNode(fdfg_op.id_op)
        self._add_node_to_graph(id_node_name, id_node)
        self._update_symbol_to_source_node_mapping(node.variable_name, id_node_name)

        if node.expression is not None:
            expression_fdfg = _convert_fhy_ast_expression_to_fdfg(node.expression)

            output_node_names = expression_fdfg.get_output_node_names()
            if len(output_node_names) > 1:
                raise RuntimeError(
                    "Expected only one output node in the expression f-DFG."
                )
            output_node_name = output_node_names[0]
            output_predecessor_node_names = expression_fdfg.predecessors(
                output_node_name
            )
            if len(output_predecessor_node_names) > 1:
                raise RuntimeError(
                    "Expected only one predecessor node for the output node in the expression f-DFG."
                )
            output_predecessor_node_name = output_predecessor_node_names[0]

            self._merge_fdfg(expression_fdfg)

            self._remove_node_from_graph(output_node_name)
            self._add_edge_to_graph(output_predecessor_node_name, id_node_name)
            self._update_symbol_to_source_node_mapping(node.variable_name, id_node_name)

    def visit_ExpressionStatement(self, node: ast.ExpressionStatement) -> None:
        # TODO: must be able to handle procedure calls

        if self._function_name is None:
            raise RuntimeError()

        def is_identifier_index(identifier: Identifier) -> bool:
            if self._symbol_table.is_symbol_defined(self._function_name, identifier):
                frame = self._symbol_table.get_frame(self._function_name, identifier)
                return isinstance(frame, VariableSymbolTableFrame) and isinstance(
                    frame.type, IndexType
                )
            else:
                raise RuntimeError(
                    f'Symbol "{identifier}" not found in the symbol table.'
                )

        index_variable_names: set[Identifier] = collect_indices(
            node.right, is_identifier_index
        )

        expression_fdfg = _convert_fhy_ast_expression_to_fdfg(node.right)

        output_node_names = expression_fdfg.get_output_node_names()
        if len(output_node_names) > 1:
            raise RuntimeError("Expected only one output node in the expression f-DFG.")
        output_node_name = output_node_names[0]
        output_predecessor_node_names = expression_fdfg.predecessors(output_node_name)
        if len(output_predecessor_node_names) > 1:
            raise RuntimeError(
                "Expected only one predecessor node for the output node in the expression f-DFG."
            )
        output_predecessor_node_name = output_predecessor_node_names[0]

        if node.left is not None:
            index_variable_names = index_variable_names.union(
                collect_indices(node.left, is_identifier_index)
            )

        if isinstance(node.left, ast.IdentifierExpression):
            symbol_name = node.left.identifier
        elif isinstance(node.left, ast.ArrayAccessExpression):
            if isinstance(node.left.array_expression, ast.IdentifierExpression):
                symbol_name = node.left.array_expression.identifier
            else:
                raise NotImplementedError(
                    f'FhYASTFunctionConverter does not support assignment to an array access on a "{type(node.left)}" expression.'
                )
        else:
            raise NotImplementedError(
                f'FhYASTFunctionConverter does not support assignment to "{type(node.left)}" expressions.'
            )

        if len(index_variable_names) > 0:
            loop_node_name_str = f"loop_{symbol_name.name_hint}_{'_'.join([index_variable_name.name_hint for index_variable_name in index_variable_names])}"
            loop_node_name = Identifier(loop_node_name_str)
            loop_node = LoopNode(index_variable_names, expression_fdfg)
            self._add_node_to_graph(loop_node_name, loop_node)
            for input_node in expression_fdfg.get_input_nodes():
                self._add_edge_to_graph(
                    self._get_source_node_name_for_symbol(input_node.symbol_name),
                    loop_node_name,
                )
            self._update_symbol_to_source_node_mapping(symbol_name, loop_node_name)
        else:
            self._merge_fdfg(expression_fdfg)
            self._remove_node_from_graph(output_node_name)
            self._update_symbol_to_source_node_mapping(
                symbol_name, output_predecessor_node_name
            )

    def visit_ReturnStatement(self, node: ast.ReturnStatement) -> None:
        # Assume one return statement at the end of the function
        expression_fdfg = _convert_fhy_ast_expression_to_fdfg(node.expression)

        output_node_names = expression_fdfg.get_output_node_names()
        if len(output_node_names) > 1:
            raise RuntimeError("Expected only one output node in the expression f-DFG.")
        output_node_name = output_node_names[0]
        output_predecessor_node_names = expression_fdfg.predecessors(output_node_name)
        if len(output_predecessor_node_names) > 1:
            raise RuntimeError(
                "Expected only one predecessor node for the output node in the expression f-DFG."
            )
        output_predecessor_node_name = output_predecessor_node_names[0]

        self._merge_fdfg(expression_fdfg)

        self._remove_node_from_graph(output_node_name)

        sink_node_name = Identifier("return")
        sink_node = SinkNode(Identifier("return"))
        self._add_node_to_graph(sink_node_name, sink_node)
        self._add_edge_to_graph(output_predecessor_node_name, sink_node_name)

    def _merge_fdfg(self, fdfg: FDFG) -> nx.MultiDiGraph:
        self._function_graph = nx.compose(self._function_graph, fdfg._graph)

        for input_node_name, input_node in zip(
            fdfg.get_input_node_names(), fdfg.get_input_nodes()
        ):
            adjacent_nodes = list(fdfg.graph.successors(input_node_name))
            for adjacent_node in adjacent_nodes:
                self._add_edge_to_graph(
                    self._get_source_node_name_for_symbol(input_node.symbol_name),
                    adjacent_node,
                )
            self._remove_node_from_graph(input_node_name)

        # for output_node_name, output_node in zip(fdfg.get_output_node_names(), fdfg.get_output_nodes()):
        #     adjacent_nodes = list(fdfg._graph.predecessors(output_node_name))
        #     for adjacent_node in adjacent_nodes:
        #         self._add_edge_to_graph(adjacent_node, self._get_source_node_name_for_symbol(output_node.symbol_name))
        #     self._remove_node_from_graph(output_node_name)

    def _get_source_node_name_for_symbol(self, symbol_name: Identifier) -> Identifier:
        return self._symbol_to_source_node_name[symbol_name]

    def _update_symbol_to_source_node_mapping(
        self, symbol_name: Identifier, source_node_name: Identifier
    ) -> None:
        self._symbol_to_source_node_name[symbol_name] = source_node_name

    def _add_edge_to_graph(
        self, source_node_name: Identifier, sink_node_name: Identifier
    ) -> None:
        self._function_graph.add_edge(source_node_name, sink_node_name)

    def _add_node_to_graph(self, node_name: Identifier, node: Node) -> None:
        self._function_graph.add_node(node_name, data=node)

    def _remove_node_from_graph(self, node_name: Identifier) -> None:
        self._function_graph.remove_node(node_name)


def _convert_fhy_ast_function_to_fdfg(
    node: ast.Function, symbol_table: SymbolTable
) -> FDFG:
    converter = FhYASTFunctionConverter(symbol_table)
    converter(node)
    graph = FDFG()
    graph.graph = converter._function_graph
    return graph
