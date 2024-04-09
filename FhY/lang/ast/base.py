# TODO Jason: Add docstring
from abc import ABC, abstractmethod
from ..span import Span

from typing import List


class ASTNode(ABC):
    """Core Abstract AST Node"""
    _span: Span

    @classmethod
    def keyname(cls) -> str:
        """Class Node Name"""
        if hasattr(cls, "__qualname__"):
            return cls.__qualname__
        return cls.__name__

    @abstractmethod
    def visit_attrs(self) -> List[str]:
        return []


class Identifier(ASTNode):
    """Abstracted Identifier Node for Providing a Unique ID.

    Args:
        _id (int): Unique Identifier
        _name_hint (str): Variable Name

    """
    # TODO Jason: Add docstring
    def __init__(self, _id: int, _name_hint: str) -> None:
        super().__init__()
        self._id = _id
        self._name_hint = _name_hint

    def visit_attrs(self) -> List[str]:
        return super().visit_attrs()

    # TODO Jason: Implement the functionality of this class
    # TODO Jason: Resolve how this identifier class can handle identifiers used in different scopes


class Component(ASTNode, ABC):
    """Component Node"""
    def __init__(self, _name: Identifier) -> None:
        super().__init__()
        self._name = _name

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        return ["_name"]


class Statement(ASTNode, ABC):
    """Abstract Statement Node"""

    def visit_attrs(self) -> List[str]:
        return super().visit_attrs()


class Expression(ASTNode, ABC):
    """Abstract Expression Definition"""

    def visit_attrs(self) -> List[str]:
        return super().visit_attrs()
