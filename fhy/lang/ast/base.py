# TODO Jason: Add docstring
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from fhy.ir import Expression as IRExpression
from fhy.ir import Identifier

from ..span import Span


class ASTNode(ABC):
    """Core abstract AST node."""

    _span: Span

    # TODO: is this method necessary?
    @classmethod
    def keyname(cls) -> str:
        """Class Node Name"""
        if hasattr(cls, "__qualname__"):
            return cls.__qualname__
        return cls.__name__

    @abstractmethod
    def visit_attrs(self) -> List[str]:
        # TODO Jason: Add docstring
        return []


class Component(ASTNode, ABC):
    # TODO Jason: Add docstring
    ...


@dataclass(frozen=True, kw_only=True)
class Function(Component, ABC):
    name: Identifier

    def visit_attrs(self) -> List[str]:
        return ["name"]


@dataclass(frozen=True)
class Module(ASTNode):
    # TODO Jason: Add docstring
    components: List[Component] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        return ["components"]


@dataclass(frozen=True, kw_only=True)
class Statement(ASTNode, ABC):
    # TODO Jason: Add docstring
    ...


class Expression(ASTNode, IRExpression, ABC):
    # TODO Jason: Add docstring
    ...
