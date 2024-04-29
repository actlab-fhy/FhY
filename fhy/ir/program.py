from typing import Any, Dict, Set
from .identifier import Identifier
from .table import SymbolTableFrame, SymbolTable, Table


class Program(object):
    _symbol_table: SymbolTable

    def __init__(self) -> None:
        self._symbol_table = SymbolTable()
