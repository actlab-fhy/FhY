"""Defines Core Abstract ASTNodes of FhY Language Grammar Constructs.

Core Nodes:
    Root: Project Root Node
    Module: Module Node

Core Abstract Nodes:
    Component: Base Components
    Function: Base Function Node
    Statement: Base Statement Node

"""

from abc import ABC
from dataclasses import dataclass, field
from typing import List

from fhy.ir import Expression as IRExpression
from fhy.ir import Identifier

from .base import ASTNode


@dataclass(frozen=True, kw_only=True)
class Root(ASTNode):
    """FhY project root ASTNode containing References to modules as children.

    Args:
        modules (List[Component]): list of modules available to project root

    """

    modules: List["Module"] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["modules"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class Module(ASTNode):
    """FhY Module ASTNode, containing references to available components.

    Args:
        components (List[Component]):

    """

    name: Identifier = field(default=Identifier("module"))
    components: List["Component"] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["components"])
        return attrs


class Component(ASTNode, ABC):
    """Abstract FhY Component ASTNode"""

    ...


@dataclass(frozen=True, kw_only=True)
class Function(Component, ABC):
    """Abstract FhY Function Component ASTNode"""

    name: Identifier

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["name"])
        return attrs


class Statement(ASTNode, ABC):
    """Abstract Statement ASTNode"""

    ...


class Expression(ASTNode, IRExpression, ABC):
    """Abstract Expression ASTNode + ir.Expression Node"""

    ...
