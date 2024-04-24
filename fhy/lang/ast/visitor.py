"""A Simple Visitor Pattern base class to visit ASTNode objects.

Classes:
    BasePass: abstract visitor pattern class
    Visitor:

"""

from abc import ABC
from typing import Any, Callable, Sequence, Union

from fhy import ir

from .base import ASTNode
from .component import Argument, Operation, Procedure
from .core import Module
from .expression import (
    ArrayAccessExpression,
    BinaryExpression,
    FloatLiteral,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    TernaryExpression,
    TupleAccessExpression,
    TupleExpression,
    UnaryExpression,
)
from .qualified_type import QualifiedType
from .statement import (
    DeclarationStatement,
    ExpressionStatement,
    ForAllStatement,
    ReturnStatement,
    SelectionStatement,
)

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
    """Abstract Visitor Pattern Class for AST Node relevant objects

    Args:
        is_recursive (bool): If true, recursively visit child nodes.

    """

    _is_recursive: bool

    def __init__(self, is_recursive: bool = True) -> None:
        self._is_recursive = is_recursive

    def __call__(self, node: ASTObject, *args: Any, **kwargs: Any) -> Any:
        return self.visit(node)

    def visit(self, node: ASTObject) -> Any:
        """A unified entry point that determines how to visit an AST object node"""
        method: Callable[[ASTObject], Any] = self.default

        for cls in type(node).mro():
            if issubclass(cls, ASTObject):
                name: str = "visit_" + cls.__name__
                if hasattr(self, name):
                    method = getattr(self, name)

        return method(node)

    def default(self, node: ASTObject) -> Any:
        """Default node visiting method"""
        raise NotImplementedError("Default Visiting Method has not been Defined.")


class Visitor(BasePass):
    """ASTObject Visitor Pattern Class"""

    def visit(self, node: ASTObject) -> Any:
        super().visit(node)

    def visit_Module(self, node: Module) -> Any:
        self.visit_sequence(node.components)

    def visit_Operation(self, node: Operation) -> Any:
        self.visit_sequence(node.args)
        self.visit(node.return_type)
        self.visit_sequence(node.body)

    def visit_Procedure(self, node: Procedure) -> Any:
        self.visit_sequence(node.args)
        self.visit_sequence(node.body)

    def visit_Argument(self, node: Argument) -> Any:
        self.visit(node.qualified_type)
        if node.name is not None:
            self.visit(node.name)

    def visit_DeclarationStatement(self, node: DeclarationStatement) -> Any:
        self.visit(node.variable_name)
        self.visit(node.variable_type)
        if node.expression is not None:
            self.visit(node.expression)

    def visit_ExpressionStatement(self, node: ExpressionStatement) -> Any:
        if node.left is not None:
            self.visit(node.left)
        self.visit(node.right)

    def visit_SelectionStatement(self, node: SelectionStatement) -> Any:
        self.visit(node.condition)
        self.visit_sequence(node.true_body)
        self.visit_sequence(node.false_body)

    def visit_ForAllStatement(self, node: ForAllStatement) -> Any:
        self.visit(node.index)
        self.visit_sequence(node.body)

    def visit_ReturnStatement(self, node: ReturnStatement) -> Any:
        self.visit(node.expression)

    def visit_UnaryExpression(self, node: UnaryExpression) -> Any:
        self.visit(node.expression)

    def visit_BinaryExpression(self, node: BinaryExpression) -> Any:
        self.visit(node.left)
        self.visit(node.right)

    def visit_TernaryExpression(self, node: TernaryExpression) -> Any:
        self.visit(node.condition)
        self.visit(node.true)
        self.visit(node.false)

    def visit_FunctionExpression(self, node: FunctionExpression) -> Any:
        self.visit(node.function)
        self.visit_sequence(node.template_types)
        self.visit_sequence(node.indices)
        self.visit_sequence(node.args)

    def visit_ArrayAccessExpression(self, node: ArrayAccessExpression) -> Any:
        self.visit(node.array_expression)
        self.visit_sequence(node.indices)

    def visit_TupleExpression(self, node: TupleExpression) -> Any:
        self.visit_sequence(node.expressions)

    def visit_TupleAccessExpression(self, node: TupleAccessExpression) -> Any:
        self.visit_TupleExpression(node.tuple_expression)
        self.visit_IntLiteral(node.element_index)

    def visit_IdentifierExpression(self, node: IdentifierExpression) -> Any:
        self.visit(node.identifier)

    def visit_IntLiteral(self, node: IntLiteral) -> Any: ...

    def visit_FloatLiteral(self, node: FloatLiteral) -> Any: ...

    def visit_QualifiedType(self, node: QualifiedType) -> Any:
        self.visit(node.base_type)
        self.visit(node.type_qualifier)

    def visit_NumericalType(self, numerical_type: ir.NumericalType) -> Any:
        self.visit(numerical_type.data_type)
        self.visit_sequence(numerical_type.shape)

    def visit_IndexType(self, index_type: ir.IndexType) -> Any:
        self.visit(index_type.lower_bound)
        self.visit(index_type.upper_bound)
        if index_type.stride is not None:
            self.visit(index_type.stride)

    def visit_Identifier(self, identifier: ir.Identifier) -> Any: ...

    def visit_sequence(self, nodes: Sequence[ASTObject]) -> Any:
        for node in nodes:
            self.visit(node)
