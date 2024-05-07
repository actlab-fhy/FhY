"""Defines Core Abstract ASTNodes of FhY Language Grammar Constructs.

Core Nodes:
    Module: Module Node

Core Abstract Nodes:
    Function: Base Function Node
    Statement: Base Statement Node
    Expression: Base AST Expression Node

"""

from abc import ABC
from dataclasses import dataclass, field
from typing import List

from fhy.ir import Expression as IRExpression
from fhy.ir import Identifier

from .base import ASTNode


@dataclass(frozen=True, kw_only=True)
class Module(ASTNode):
    """FhY Module ASTNode, containing references to available statements.

    Args:
        statements (List[Statement]):

    """

    name: Identifier = field(default=Identifier("module"))
    statements: List["Statement"] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["statements"])
        return attrs


class Statement(ASTNode, ABC):
    """Abstract Statement ASTNode."""


@dataclass(frozen=True, kw_only=True)
class Function(Statement, ABC):
    """Abstract FhY Function Component ASTNode."""

    name: Identifier

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["name"])
        return attrs


class Expression(ASTNode, IRExpression, ABC):
    """Abstract Expression ASTNode + ir.Expression Node."""
