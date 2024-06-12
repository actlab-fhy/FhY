from typing import Any, List, Optional, Callable, Union

from fhy.lang.ast.node.core import Statement
from ..visitor import Transformer, Statements, BasePass
from fhy.ir.table import SymbolTable, SymbolTableFrame, VariableSymbolTableFrame
from fhy.lang.ast.node.core import Expression, Module
from fhy.ir.identifier import Identifier
from fhy.lang.ast.node.qualified_type import QualifiedType
from fhy.lang.ast.node.statement import (
    DeclarationStatement,
    ExpressionStatement,
    SelectionStatement,
    ForAllStatement,
    ReturnStatement,
    Procedure,
    Operation,
)
from fhy.lang.ast.node.expression import (
    IdentifierExpression,
    UnaryExpression,
    BinaryExpression,
    TernaryExpression,
    FunctionExpression,
    ArrayAccessExpression,
    IntLiteral,
    FloatLiteral,
    TupleAccessExpression,
    TupleExpression,
)
from fhy.utils import Stack
from .index_collector import collect_indices
from fhy.ir.type import TypeQualifier, NumericalType, PrimitiveDataType, DataType, IndexType


# TODO: refactor out into the AST node definition classes?
AtomicExpression = Union[
    IntLiteral,
    FloatLiteral,
    IdentifierExpression,
    ArrayAccessExpression,
    TupleAccessExpression,
]


# TODO: come up with a better naming scheme for temporary symbols for debugging
class IndividualExpressionDecomposer(BasePass):
    _get_symbol_table_frame: Callable[[Identifier], SymbolTableFrame]

    _statements: List[Statement]

    def __init__(
        self, get_symbol_table_frame_func: Callable[[Identifier], SymbolTableFrame]
    ) -> None:
        super().__init__()
        self._get_symbol_table_frame = get_symbol_table_frame_func
        self._statements = []

    @property
    def statements(self) -> List[Statement]:
        return self._statements

    def __call__(self, node: Expression) -> None:
        if not isinstance(node, Expression):
            raise RuntimeError(f"{__class__.__name__} expects an expression node.")
        return super().__call__(node)

    def visit(self, node: Any) -> AtomicExpression:
        return super().visit(node)

    def visit_UnaryExpression(self, node: UnaryExpression) -> IdentifierExpression:
        sub_expression = self.visit(node.expression)
        return self._add_intermediate_expression(
            UnaryExpression(operation=node.operation, expression=sub_expression)
        )

    def visit_BinaryExpression(self, node: BinaryExpression) -> IdentifierExpression:
        left_expression = self.visit(node.left)
        right_expression = self.visit(node.right)
        return self._add_intermediate_expression(
            BinaryExpression(
                operation=node.operation, left=left_expression, right=right_expression
            )
        )

    def visit_TernaryExpression(self, node: TernaryExpression) -> IdentifierExpression:
        condition_expression = self.visit(node.condition)
        true_expression = self.visit(node.true)
        false_expression = self.visit(node.false)
        return self._add_intermediate_expression(
            TernaryExpression(
                condition=condition_expression,
                true=true_expression,
                false=false_expression,
            )
        )

    def visit_ArrayAccessExpression(
        self, node: ArrayAccessExpression
    ) -> ArrayAccessExpression:
        array_expression = self.visit(node.array_expression)
        # TODO: need to copy the indices here, but cannot visit because they are not data expressions
        return ArrayAccessExpression(
            array_expression=array_expression, indices=node.indices
        )

    def visit_IdentifierExpression(
        self, node: IdentifierExpression
    ) -> IdentifierExpression:
        return IdentifierExpression(identifier=node.identifier)

    def _collect_indices(self, node: Expression) -> list[Identifier]:
        def is_identifier_index_func(name: Identifier) -> bool:
            frame = self._get_symbol_table_frame(name)
            return isinstance(frame, VariableSymbolTableFrame) and isinstance(frame.type, IndexType)

        return collect_indices(node, is_identifier_index_func)

    def _add_intermediate_expression(
        self, expression: Expression
    ) -> IdentifierExpression:
        intermediate_symbol = Identifier("t")
        symbol_declaration_statement = DeclarationStatement(
            variable_name=intermediate_symbol,
            # TODO: Resolve the type based on the expression!
            variable_type=QualifiedType(
                type_qualifier=TypeQualifier.TEMP,
                base_type=NumericalType(DataType(PrimitiveDataType.INT32)),
            ),
        )
        symbol_assignment_statement = ExpressionStatement(
            left=IdentifierExpression(identifier=intermediate_symbol), right=expression
        )
        self._statements.append(symbol_declaration_statement)
        self._statements.append(symbol_assignment_statement)
        return IdentifierExpression(identifier=intermediate_symbol)


def _decompose_individual_expression(
    expression: Expression,
    get_symbol_table_frame_func: Callable[[Identifier], SymbolTableFrame],
) -> List[Statement]:
    decomposer = IndividualExpressionDecomposer(get_symbol_table_frame_func)
    decomposer(expression)
    return decomposer.statements


class ExpressionDecomposer(Transformer):
    _symbol_table: SymbolTable
    _namespace_stack: Stack[Identifier]

    def __init__(self, symbol_table: SymbolTable) -> None:
        super().__init__()
        self._symbol_table = symbol_table
        self._namespace_stack = Stack()

    def visit_Module(self, node: Module) -> Module:
        self._namespace_stack.push(node.name)
        ret = super().visit_Module(node)
        self._namespace_stack.pop()
        return ret

    def visit_Operation(self, node: Operation) -> Statements:
        self._namespace_stack.push(node.name)
        ret = super().visit_Operation(node)
        self._namespace_stack.pop()
        return ret

    def visit_Procedure(self, node: Procedure) -> Statements:
        self._namespace_stack.push(node.name)
        ret = super().visit_Procedure(node)
        self._namespace_stack.pop()
        return ret

    def visit_Statement(self, node: Statement) -> Statements:
        if isinstance(node, DeclarationStatement):
            return self.visit_DeclarationStatement(node)
        elif isinstance(node, ExpressionStatement):
            if isinstance(node.right, FunctionExpression):
                return node.right

            intermediate_statements = _decompose_individual_expression(
                node.right,
                lambda i: self._symbol_table.get_frame(self._namespace_stack.peek(), i),
            )
            final_intermediate_expression = intermediate_statements[-1]
            if not isinstance(final_intermediate_expression, ExpressionStatement):
                raise RuntimeError()
            if not isinstance(final_intermediate_expression.left, IdentifierExpression):
                raise RuntimeError()
            result_identifier = final_intermediate_expression.left.identifier

            return intermediate_statements + [
                ExpressionStatement(
                    left=node.left,
                    right=IdentifierExpression(identifier=result_identifier),
                )
            ]
        elif isinstance(node, SelectionStatement):
            return self.visit_SelectionStatement(node)
        elif isinstance(node, ForAllStatement):
            return self.visit_ForAllStatement(node)
        elif isinstance(node, ReturnStatement):
            if isinstance(node.expression, FunctionExpression):
                return node.expression

            intermediate_statements = _decompose_individual_expression(
                node.expression,
                lambda i: self._symbol_table.get_frame(self._namespace_stack.peek(), i),
            )
            final_intermediate_expression = intermediate_statements[-1]
            if not isinstance(final_intermediate_expression, ExpressionStatement):
                raise RuntimeError()
            if not isinstance(final_intermediate_expression.left, IdentifierExpression):
                raise RuntimeError()
            result_identifier = final_intermediate_expression.left.identifier

            return intermediate_statements + [
                ReturnStatement(
                    expression=IdentifierExpression(identifier=result_identifier)
                )
            ]
        else:
            return super().visit_Statement(node)


# Update the typing here to include that whatever of the three you pass you will get the same back
def decompose_expressions(
    ast_node: Union[Module, Procedure, Operation], symbol_table: SymbolTable
) -> Union[Module, Procedure, Operation]:
    decomposer = ExpressionDecomposer(symbol_table)
    return decomposer(ast_node)
