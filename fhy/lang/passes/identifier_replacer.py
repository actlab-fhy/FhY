from copy import copy
from typing import Dict
from fhy.lang.ast import Transformer
from fhy.utils.alias import ASTObject
from fhy import ir


class IdentifierReplacer(Transformer):
    _identifier_map: Dict[ir.Identifier, ir.Identifier]

    def __init__(self, identifier_map: Dict[ir.Identifier, ir.Identifier]):
        super().__init__()
        self._identifier_map = identifier_map

    def visit_Identifier(self, identifier: ir.Identifier) -> ir.Identifier:
        return copy(self._identifier_map.get(identifier, identifier))


def replace_identifiers(node: ASTObject, identifier_map: Dict[ir.Identifier, ir.Identifier]) -> ASTObject:
    return IdentifierReplacer(identifier_map).visit(node)
