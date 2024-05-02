"""Visitor Passes to validate and analyze AST Semantics.

Modules:
    identifier_collector:
    symbol_table_builder:

Passes
    collect_identifiers:
    build_symbol_table:

"""

from .identifier_collector import collect_identifiers
from .symbol_table_builder import build_symbol_table
