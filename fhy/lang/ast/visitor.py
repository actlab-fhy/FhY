"""
FhY/lang/visitor.py

"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Generator, Tuple, Union

from .base import ASTNode
from .component import Native, Operation, Procedure
from .core import Module
from .expression import BinaryExpression, IdentifierExpression, UnaryExpression
from .qualified_type import QualifiedType
from .statement import DeclarationStatement, ExpressionStatement, ReturnStatement

from fhy import ir


ASTObject = Union[ASTNode, ir.Identifier, ir.Type, ir.DataType]


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


# TODO: not sure if this is necessary
class Visitor(BasePass):
    """Base visitor pattern for the FhY AST. Visits depth-first.

    """

    def visit(self, node: ASTObject) -> None:
        super().visit(node)
