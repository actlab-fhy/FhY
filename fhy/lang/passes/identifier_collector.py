from typing import Set

from fhy import ir
from fhy.lang import ast
from fhy.lang.ast.visitor import ASTObject


class IdentifierCollector(ast.Visitor):
    """Collects all identifiers in the AST for any given node"""

    _identifers: Set[ir.Identifier]

    def __init__(self) -> None:
        super().__init__()
        self._identifers = set()

    @property
    def identifiers(self) -> Set[ir.Identifier]:
        return self._identifers

    def visit_Identifier(self, identifier: ir.Identifier) -> None:
        self._identifers.add(identifier)


def collect_identifiers(node: ASTObject) -> Set[ir.Identifier]:
    collector = IdentifierCollector()
    collector(node)
    return collector.identifiers
