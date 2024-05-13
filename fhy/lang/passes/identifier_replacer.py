"""Identifier Replacement Transformer."""

from copy import copy
from typing import Dict

from fhy import ir
from fhy.lang.ast import Transformer
from fhy.utils.alias import ASTObject


class IdentifierReplacer(Transformer):
    """Replace Identifiers.

    Args:
        identifier_map (Dict[ir.Identifier, ir.Identifier]): _description_

    Returns:
        _type_: _description_

    """

    _identifier_map: Dict[ir.Identifier, ir.Identifier]

    def __init__(self, identifier_map: Dict[ir.Identifier, ir.Identifier]):
        super().__init__()
        self._identifier_map = identifier_map

    def visit_Identifier(self, identifier: ir.Identifier) -> ir.Identifier:
        # NOTE: All we want to do is copy the existing Identifier?
        #       Do we not want to also bank encountered identifiers?
        # NOTE: This is a transformer by virtue of providing return values.
        return copy(self._identifier_map.get(identifier, identifier))


def replace_identifiers(
    node: ASTObject, identifier_map: Dict[ir.Identifier, ir.Identifier]
) -> ASTObject:
    """Copy Identifiers."""
    return IdentifierReplacer(identifier_map).visit(node)
