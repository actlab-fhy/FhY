"""Core AST nodes for FhY language constructs.

Core Nodes:
    Module: Module node.

Core Abstract Nodes:
    Function: Base function node.
    Statement: Base statement node.
    Expression: Base AST expression node.

"""

from abc import ABC
from dataclasses import dataclass, field
from typing import List

from fhy.ir.expression import Expression as IRExpression
from fhy.ir.identifier import Identifier as IRIdentifier

from .base import ASTNode


@dataclass(frozen=True, kw_only=True)
class Module(ASTNode):
    """FhY module AST node.

    Args:
        name (IRIdentifier): Name of the module.
        statements (List[Statement]): List of statements in the module.

    Attributes:
        name (IRIdentifier): Name of the module.
        statements (List[Statement]): List of statements in the module.

    """

    # TODO: remove default value for name and have converter create name
    name: IRIdentifier = field(default=IRIdentifier("module"))
    statements: List["Statement"] = field(default_factory=list)

    def get_visit_attrs(self) -> List[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["name", "statements"])
        return attrs


class Statement(ASTNode, ABC):
    """Abstract statement AST node."""


@dataclass(frozen=True, kw_only=True)
class Function(Statement, ABC):
    """Abstract FhY function node.

    Used as a base for the function nodes such as procedures and operations.

    Attributes:
        name (IRIdentifier): Name of the function.

    """

    name: IRIdentifier

    def get_visit_attrs(self) -> List[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["name"])
        return attrs


class Expression(ASTNode, IRExpression, ABC):
    """Abstract expression AST node.

    Also is an expression from the IR to enable use in symbol table fields.

    """
