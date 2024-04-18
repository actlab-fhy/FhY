"""
FhY/lang/visitor.py

"""

from typing import Any, Callable, Generator, Tuple

from .ast.core import ASTNode


def get_node_key(node: ASTNode) -> str:
    """Retrieves Node Key Name"""
    return node.keyname()


def get_node_name(node: object) -> str:
    """Retrieves the Name of a Class Instance."""
    cls = node.__class__
    if hasattr(cls, "__qualname__"):
        return cls.__qualname__
    return cls.__name__


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


class BaseVisitor:
    """Base Visitor Pattern Class Interface.

    Args:
        accessor (Callable[[Node], str]): Function that accepts a Node, and
            returns a string identifier name

    """

    def __init__(self, accessor: Callable[[ASTNode], str] = get_node_name) -> None:
        self.accessor = accessor

    def visit(self, node: ASTNode):
        """Visit a Node using Node Keyname"""
        name = "visit_" + self.accessor(node)
        if hasattr(self, name):
            method = getattr(self, name)
        else:
            method = self.generic
        return method(node)

    def generic(self, node: ASTNode):
        """Fallback generic visiting Method"""
        for child in iter_children(node):
            self.visit(child)
