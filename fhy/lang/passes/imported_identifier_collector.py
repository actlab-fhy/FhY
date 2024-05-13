from typing import Set

from fhy import ir
from fhy.lang import ast
from fhy.utils.alias import ASTObject


class ImportedIdentifierCollector(ast.Visitor):
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
    collector = ImportedIdentifierCollector()
    collector(node)
    return collector.identifiers
