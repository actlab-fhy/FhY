"""
FhY/lang/visitor.py

"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Generator, Tuple

from .base import ASTNode
from .component import Native, Operation, Procedure
from .core import Module
from .expression import BinaryExpression, IdentifierExpression, UnaryExpression
from .qualified_type import QualifiedType
from .statement import DeclarationStatement, ExpressionStatement, ReturnStatement


def iter_fields(node: ASTNode) -> Generator[Tuple[str, Any], None, None]:
    """Iterates through Relevant Attributes of a Node.

    Returns:
        Tuple[str, Any]

    """
    for field in node.visit_attrs():
        if not hasattr(node, field):
            continue
        yield field, getattr(node, field)


def iter_children(node: ASTNode) -> Generator[ASTNode, None, None]:
    """Yields all Direct Child Nodes"""
    for _, field in iter_fields(node):
        if isinstance(field, ASTNode):
            yield field
        elif isinstance(field, list):
            for child in field:
                if isinstance(child, ASTNode):
                    yield child


class BasePass(ABC):
    @abstractmethod
    def __call__(self, node: ASTNode, *args: Any, **kwargs: Any) -> Any:
        ...


class Visitor(BasePass):
    """Base visitor pattern for the FhY AST. Visits depth-first.

    """
    _is_recursive: bool

    def __init__(self, is_recursive: bool = True) -> None:
        self._is_recursive = is_recursive

    def __call__(self, node: ASTNode) -> None:
        self.visit(node)

    def visit(self, node: ASTNode) -> None:
        if self._is_recursive:
            for child in iter_children(node):
                self.visit(child)

                name: str = "visit_" + node.keyname()
            if hasattr(self, name):
                method: Callable[[ASTNode], None] = getattr(self, name)
            else:
                method: Callable[[ASTNode], None] = self.default

            method(node)
        else:
            raise NotImplementedError()

    def default(self, node: ASTNode) -> None:
        """Default visitor method"""

    def visit_module(self, module: Module) -> None:
        """Visit a Module node"""

    def visit_native(self, native: Native) -> None:
        """Visit a Native node"""

    def visit_operation(self, operation: Operation) -> None:
        """Visit an Operation node"""

    def visit_procedure(self, procedure: Procedure) -> None:
        """Visit a Procedure node"""

    def visit_declaration_statement(self, statement: DeclarationStatement) -> None:
        """Visit a Declaration Statement node"""

    def visit_expression_statement(self, statement: ExpressionStatement) -> None:
        """Visit an Expression Statement node"""

    def visit_return_statement(self, statement: ReturnStatement) -> None:
        """Visit a Return Statement node"""

    def visit_binary_expression(self, expression: BinaryExpression) -> None:
        """Visit a Binary Expression node"""

    def visit_identifier_expression(self, expression: IdentifierExpression) -> None:
        """Visit an Identifier Expression node"""

    def visit_unary_expression(self, expression: UnaryExpression) -> None:
        """Visit a Unary Expression node"""

    def visit_qualified_type(self, qualified_type: QualifiedType) -> None:
        """Visit a Qualified Type node"""
