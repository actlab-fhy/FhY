"""AST Visitor Pass to collect Import Identifiers."""

from typing import Set

from fhy import ir
from fhy.lang.ast import node as ast
from fhy.lang.ast.alias import ASTObject
from fhy.lang.ast.visitor import Visitor


class ImportedIdentifierCollector(Visitor):
    """Visitor pass to Collect Import identifiers from AST Nodes."""

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
    """Collect all identifiers from Import Statements from a given node.

    Args:
        node (ASTObject): AST Node object.

    Returns:
        Set[ir.Identifier]: Set of discovered Import Identifiers from Node graph.

    """
    collector = ImportedIdentifierCollector()
    collector(node)

    return collector.identifiers
