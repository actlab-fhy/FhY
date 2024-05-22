"""Identifier replacement transformer."""

from copy import copy
from typing import Dict

from fhy import ir
from fhy.lang.ast.alias import ASTObject
from fhy.lang.ast.visitor import Transformer


class IdentifierReplacer(Transformer):
    """Replace identifiers.

    Args:
        identifier_map (Dict[ir.Identifier, ir.Identifier]): mapping describing
            identifiers to change from and to.

    """

    _identifier_map: Dict[ir.Identifier, ir.Identifier]

    def __init__(self, identifier_map: Dict[ir.Identifier, ir.Identifier]):
        super().__init__()
        self._identifier_map = identifier_map

    def visit_Identifier(self, identifier: ir.Identifier) -> ir.Identifier:
        return copy(self._identifier_map.get(identifier, identifier))


def replace_identifiers(
    node: ASTObject, identifier_map: Dict[ir.Identifier, ir.Identifier]
) -> ASTObject:
    """Replace identifiers within AST.

    Args:
        node (ASTObject): AST object node.
        identifier_map (Dict[ir.Identifier, ir.Identifier]): mapping describing
            identifiers to change from and to.

    Returns:
        (ASTObject): node with identifiers replaced as prescribed by mapping.

    """
    return IdentifierReplacer(identifier_map).visit(node)
