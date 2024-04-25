from typing import Any, Dict, Set
from .module import Module
from .table import Table, SymbolTableFrame
from .identifier import Identifier


class Program(object):
    _modules: Dict[Identifier, Module]
    _symbol_tables: Dict[Identifier, Table[Identifier, SymbolTableFrame]]
    # _namespaces: Dict[Identifier, Table[Identifier, NamespaceTableFrame]]

    def __init__(self) -> None:
        self._modules = {}
        self._symbol_tables = {}

    def add_module(self, module: Module) -> None:
        self._modules[module.name] = module
        # TODO: use a pass to get all the symbols from the modules?

    @property
    def modules(self) -> Set[Module]:
        return set(self._modules.values())

    # @property
    # def namespaces(self) -> Set[Identifier]:
    #     return set(self._namespaces.values())
