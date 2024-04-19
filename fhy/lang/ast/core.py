# TODO Jason: Add docstring
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fhy.ir import Expression as IRExpression
from fhy.ir import Identifier

from .base import ASTNode
from .directory import register_ast_node


@register_ast_node
@dataclass(frozen=True)
class Module(ASTNode):
    # TODO Jason: Add docstring
    components: List["Component"] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        return ["components"]


class Component(ASTNode, ABC):
    # TODO Jason: Add docstring
    ...


@dataclass(frozen=True, kw_only=True)
class Function(Component, ABC):
    name: Identifier

    def visit_attrs(self) -> List[str]:
        return ["name"]


@dataclass(frozen=True, kw_only=True)
class Statement(ASTNode, ABC):
    # TODO Jason: Add docstring
    ...


@dataclass(frozen=True, kw_only=True)
class Expression(ASTNode, IRExpression, ABC):
    # TODO Jason: Add docstring
    ...
