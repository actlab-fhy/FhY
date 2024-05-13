"""Visitor Passes to validate and analyze AST Semantics.

Modules:
    identifier_collector:
    symbol_table_builder:

Passes
    collect_identifiers:
    build_symbol_table:

"""

from .identifier_collector import collect_identifiers
from .identifier_replacer import replace_identifiers
from .imported_identifier_collector import collect_imported_identifiers
from .symbol_table_builder import build_symbol_table
