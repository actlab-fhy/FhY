# TODO Jason: Add docstring
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from fhy.ir import Expression as IRExpression
from fhy.ir import Identifier
from ..span import Span



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


class Component(ASTNode, ABC):
    """Component Node"""
    def __init__(self, _name: Identifier) -> None:
        super().__init__()
        self._name = _name

    @property
    def name(self) -> Identifier:
        return self._name

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        return ["_name"]


@dataclass(frozen=True)
class Module(object):
    components: List[Component]


class Statement(ASTNode, ABC):
    """Abstract Statement Node"""

    def visit_attrs(self) -> List[str]:
        return super().visit_attrs()


class Expression(ASTNode, IRExpression, ABC):
    """Abstract Expression Definition"""

    def visit_attrs(self) -> List[str]:
        return super().visit_attrs()
