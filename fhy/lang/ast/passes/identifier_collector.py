"""Simple Visitor AST Pass to collect Symbol Identifiers.

Functions:
    collect_identifiers: Primary API Entry point to collect Identifiers from AST Node.

Classes:
    IdentifierCollector: The workhorse driving the collection of identifiers.

"""

from typing import Set

from fhy import ir
from fhy.lang.ast.alias import ASTObject
from fhy.lang.ast.visitor import Visitor


# TODO: The following pass should use a Listener pattern instead, but is blocked
#       by current implementation of listener.
class IdentifierCollector(Visitor):
    """Collect all identifiers in the AST for any given node."""

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
    """Return a set of Identifier objects from a given AST Node object."""
    collector = IdentifierCollector()
    collector(node)
    return collector.identifiers
