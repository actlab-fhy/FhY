"""Program Root Node."""

from typing import Dict, Union

# from fhy.lang.ast.core import Module as ASTModule
from .table import SymbolTable
from .identifier import Identifier


class Program(object):
    """Program Object."""

    # _components: Dict[Identifier, Union[ASTModule]]
    _symbol_table: SymbolTable

    def __init__(self) -> None:
        self._components = {}
        self._symbol_table = SymbolTable()
