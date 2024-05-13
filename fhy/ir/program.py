"""Program Root Node."""

# from fhy.lang.ast.core import Module as ASTModule
from .table import SymbolTable


class Program(object):
    """Program Object."""

    _components: dict  # Dict[Identifier, Union[ASTModule]]
    _symbol_table: SymbolTable

    def __init__(self) -> None:
        self._components = {}
        self._symbol_table = SymbolTable()
