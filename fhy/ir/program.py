"""Program Root Node."""

from .table import SymbolTable


class Program(object):
    """Program Object."""

    _symbol_table: SymbolTable

    def __init__(self) -> None:
        self._symbol_table = SymbolTable()
