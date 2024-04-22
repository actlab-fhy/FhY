"""
FhY/lang/visitor.py

"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Generator, Tuple, Union, Sequence

from .base import ASTNode
from .component import Native, Operation, Procedure, Argument
from .core import Module
from .expression import BinaryExpression, IdentifierExpression, UnaryExpression, TernaryExpression, TupleAccessExpression, TupleExpression, FunctionExpression, ArrayAccessExpression, IntLiteral, FloatLiteral
from .qualified_type import QualifiedType
from .statement import DeclarationStatement, ExpressionStatement, ReturnStatement, SelectionStatement, ForAllStatement

from fhy import ir


ASTObject = Union[ASTNode, ir.Identifier, ir.Type, ir.DataType]


# def iter_fields(node: ASTNode) -> Generator[Tuple[str, Any], None, None]:
#     """Iterates through Relevant Attributes of a Node.

#     Returns:
#         Tuple[str, Any]

#     """
#     for field in node.visit_attrs():
#         if not hasattr(node, field):
#             continue
#         yield field, getattr(node, field)


# def iter_children(node: ASTNode) -> Generator[ASTNode, None, None]:
#     """Yields all Direct Child Nodes"""
#     for _, field in iter_fields(node):
#         if isinstance(field, ASTNode):
#             yield field
#         elif isinstance(field, list):
#             for child in field:
#                 if isinstance(child, ASTNode):
#                     yield child


class BasePass(ABC):
    _is_recursive: bool

    def __init__(self, is_recursive: bool = True) -> None:
        self._is_recursive = is_recursive

    def __call__(self, node: ASTObject, *args: Any, **kwargs: Any) -> Any:
        return self.visit(node)

    def visit(self, node: ASTObject) -> Any:
        method: Callable[[ASTObject], Any] = self.default
        for cls in type(node).mro():
            if issubclass(cls, ASTObject):
                name: str = "visit_" + cls.__name__
                if hasattr(self, name):
                    method = getattr(self, name)
        return method(node)

    def default(self, node: ASTObject) -> Any:
        pass


class Visitor(BasePass):
    def visit(self, node: ASTObject) -> None:
        super().visit(node)

    def visit_Module(self, node: Module) -> None:
        self.visit_sequence(node.components)

    def visit_Operation(self, node: Operation) -> None:
        self.visit_sequence(node.args)
        self.visit(node.return_type)
        self.visit_sequence(node.body)

    def visit_Procedure(self, node: Procedure) -> None:
        self.visit_sequence(node.args)
        self.visit_sequence(node.body)

    def visit_Argument(self, node: Argument) -> None:
        self.visit(node.type)
        if node.name is not None:
            self.visit(node.name)

    def visit_DeclarationStatement(self, node: DeclarationStatement) -> None:
        self.visit(node.variable_name)
        self.visit(node.variable_type)
        if node.expression is not None:
            self.visit(node.expression)

    def visit_ExpressionStatement(self, node: ExpressionStatement) -> None:
        if node.left is not None:
            self.visit(node.left)
        self.visit(node.right)

    def visit_SelectionStatement(self, node: SelectionStatement) -> None:
        self.visit(node.condition)
        self.visit_sequence(node.true_body)
        self.visit_sequence(node.false_body)

    def visit_ForAllStatement(self, node: ForAllStatement) -> None:
        self.visit(node.index)
        self.visit_sequence(node.body)

    def visit_ReturnStatement(self, node: ReturnStatement) -> None:
        self.visit(node.expression)

    def visit_UnaryExpression(self, node: UnaryExpression) -> None:
        self.visit(node.expression)

    def visit_BinaryExpression(self, node: BinaryExpression) -> None:
        self.visit(node.left)
        self.visit(node.right)

    def visit_TernaryExpression(self, node: TernaryExpression) -> None:
        self.visit(node.condition)
        self.visit(node.true)
        self.visit(node.false)

    def visit_FunctionExpression(self, node: FunctionExpression) -> None:
        self.visit(node.function)
        self.visit_sequence(node.template_types)
        self.visit_sequence(node.indices)
        self.visit_sequence(node.args)

    def visit_ArrayAccessExpression(self, node: ArrayAccessExpression) -> None:
        self.visit(node.array_expression)
        self.visit_sequence(node.indices)

    def visit_TupleExpression(self, node: TupleExpression) -> None:
        self.visit_sequence(node.expressions)

    def visit_IdentifierExpression(self, node: IdentifierExpression) -> None:
        self.visit(node.identifier)

    def visit_IntLiteral(self, node: IntLiteral) -> None:
        pass

    def visit_FloatLiteral(self, node: FloatLiteral) -> None:
        pass

    def visit_QualifiedType(self, node: QualifiedType) -> None:
        self.visit(node.base_type)

    def visit_NumericalType(self, numerical_type: ir.NumericalType) -> None:
        self.visit(numerical_type.data_type)
        self.visit_sequence(numerical_type.shape)

    def visit_IndexType(self, index_type: ir.IndexType) -> None:
        self.visit(index_type.lower_bound)
        self.visit(index_type.upper_bound)
        if index_type.stride is not None:
            self.visit(index_type.stride)

    def visit_Identifier(self, identifier: ir.Identifier) -> None:
        pass

    def visit_sequence(self, nodes: Sequence[ASTObject]) -> None:
        for node in nodes:
            self.visit(node)
