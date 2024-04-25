"""A Simple Visitor Pattern base class to visit ASTNode objects.

Classes:
    BasePass: abstract visitor pattern class
    Visitor:
    Listener:

"""

from abc import ABC
from typing import Any, Callable, Sequence, Union

from fhy import ir

from ..span import Source, Span
from .base import ASTNode
from .component import Argument, Import, Operation, Procedure
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

ASTObject = Union[ASTNode, ir.Identifier, ir.Type, ir.DataType, ir.TypeQualifier]


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


def get_cls_name(obj: Any) -> str:
    """Retrieves the Class name of an object instance."""
    if not hasattr(obj, "keyname"):
        return obj.__class__.__name__

    return obj.keyname()


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
        name = f"visit_{get_cls_name(node)}"
        method: Callable[[ASTObject], Any] = getattr(self, name, self.default)

        return method(node)

    def default(self, node: ASTObject) -> Any:
        """Default node visiting method"""
        raise NotImplementedError(
            f"AST node {type(node)} is not supported yet and the default method "
            "is not implemented."
        )


class Visitor(BasePass):
    """ASTObject Visitor Pattern Class"""

    def visit(self, node: ASTObject) -> None:
        super().visit(node)

    def visit_Module(self, node: Module) -> None:
        self.visit(node.name)
        self.visit_sequence(node.components)

    def visit_Import(self, node: Import) -> None:
        self.visit_sequence(node.module_path)

    def visit_Operation(self, node: Operation) -> None:
        self.visit_sequence(node.args)
        self.visit(node.return_type)
        self.visit_sequence(node.body)

    def visit_Procedure(self, node: Procedure) -> None:
        self.visit_sequence(node.args)
        self.visit_sequence(node.body)

    def visit_Argument(self, node: Argument) -> None:
        self.visit(node.qualified_type)
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

    def visit_TupleAccessExpression(self, node: TupleAccessExpression) -> None:
        self.visit_TupleExpression(node.tuple_expression)
        self.visit_IntLiteral(node.element_index)

    def visit_IdentifierExpression(self, node: IdentifierExpression) -> None:
        self.visit(node.identifier)

    def visit_IntLiteral(self, node: IntLiteral) -> None: ...

    def visit_FloatLiteral(self, node: FloatLiteral) -> None: ...

    def visit_QualifiedType(self, node: QualifiedType) -> None:
        self.visit(node.base_type)
        self.visit(node.type_qualifier)

    def visit_NumericalType(self, numerical_type: ir.NumericalType) -> None:
        self.visit(numerical_type.data_type)
        self.visit_sequence(numerical_type.shape)

    def visit_IndexType(self, index_type: ir.IndexType) -> None:
        self.visit(index_type.lower_bound)
        self.visit(index_type.upper_bound)
        if index_type.stride is not None:
            self.visit(index_type.stride)

    def visit_DataType(self, data_type: ir.DataType) -> None: ...

    def visit_TypeQualifier(self, type_qualifier: ir.TypeQualifier) -> None: ...

    def visit_Identifier(self, identifier: ir.Identifier) -> None: ...

    def visit_sequence(self, nodes: Sequence[ASTObject]) -> None:
        for node in nodes:
            self.visit(node)

    def visit_Span(self, span: Span) -> None: ...

    def visit_Source(self, source: Source) -> None: ...


class Listener(BasePass):
    """ASTObject Listener Pattern Class"""

    def default(self, node: ASTObject) -> None:
        if isinstance(node, list):
            self.visit_sequence(node)
        super().default(node)

    def visit_Module(self, node: Module) -> None: ...

    def visit_Operation(self, node: Operation) -> None: ...

    def visit_Procedure(self, node: Procedure) -> None: ...

    def visit_Argument(self, node: Argument) -> None: ...

    def visit_DeclarationStatement(self, node: DeclarationStatement) -> None: ...

    def visit_ExpressionStatement(self, node: ExpressionStatement) -> None: ...

    def visit_SelectionStatement(self, node: SelectionStatement) -> None: ...

    def visit_ForAllStatement(self, node: ForAllStatement) -> None: ...

    def visit_ReturnStatement(self, node: ReturnStatement) -> None: ...

    def visit_UnaryExpression(self, node: UnaryExpression) -> None: ...

    def visit_BinaryExpression(self, node: BinaryExpression) -> None: ...

    def visit_TernaryExpression(self, node: TernaryExpression) -> None: ...

    def visit_FunctionExpression(self, node: FunctionExpression) -> None: ...

    def visit_ArrayAccessExpression(self, node: ArrayAccessExpression) -> None: ...

    def visit_TupleExpression(self, node: TupleExpression) -> None: ...

    def visit_TupleAccessExpression(self, node: TupleAccessExpression) -> None: ...

    def visit_IdentifierExpression(self, node: IdentifierExpression) -> None: ...

    def visit_IntLiteral(self, node: IntLiteral) -> None: ...

    def visit_FloatLiteral(self, node: FloatLiteral) -> None: ...

    def visit_QualifiedType(self, node: QualifiedType) -> None: ...

    def visit_DataType(self, node: ir.DataType) -> None: ...

    def visit_NumericalType(self, numerical_type: ir.NumericalType) -> None: ...

    def visit_IndexType(self, index_type: ir.IndexType) -> None: ...

    def visit_Identifier(self, identifier: ir.Identifier) -> None: ...

    def visit_sequence(self, nodes: Sequence[ASTObject]) -> None: ...

    def visit_Span(self, span: Span) -> None: ...

    def visit_Source(self, source: Source) -> None: ...
