from typing import Set

from fhy import ir
from fhy.lang import ast
from fhy.utils.alias import ASTObject


class IdentifierCollector(ast.Visitor):
    """Collects all identifiers in the AST for any given node"""

    _identifiers: Set[ir.Identifier]

    def __init__(self) -> None:
        super().__init__()
        self._identifiers = set()

    @property
    def identifiers(self) -> Set[ir.Identifier]:
        return self._identifiers

    def visit_Identifier(self, identifier: ir.Identifier) -> None:
        self._identifiers.add(identifier)


def collect_identifiers(node: ASTObject) -> Set[ir.Identifier]:
    collector = IdentifierCollector()
    collector(node)
    return collector.identifiers
