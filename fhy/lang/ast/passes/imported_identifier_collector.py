"""AST visitor pass to collect import identifiers."""

from typing import Set

from fhy import ir
from fhy.lang.ast import node as ast
from fhy.lang.ast.alias import ASTObject
from fhy.lang.ast.visitor import Visitor


class ImportedIdentifierCollector(Visitor):
    """Visitor pass to collect import identifiers from AST nodes."""

    _identifiers: Set[ir.Identifier]

    def __init__(self) -> None:
        super().__init__()
        self._identifiers = set()

    @property
    def identifiers(self) -> Set[ir.Identifier]:
        return self._identifiers

    def visit_Import(self, node: ast.Import) -> None:
        self._identifiers.add(node.name)
        super().visit_Import(node)


def collect_imported_identifiers(node: ASTObject) -> Set[ir.Identifier]:
    """Collect all identifiers from import statements from a given node.

    Args:
        node (ASTObject): AST node object.

    Returns:
        Set[ir.Identifier]: Set of discovered import identifiers from node graph.

    """
    collector = ImportedIdentifierCollector()
    collector(node)

    return collector.identifiers
