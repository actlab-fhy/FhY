from abc import ABC
from ..node import Node
from fhy.ir.identifier import Identifier
from ..core import FDFG


class ParametricNode(Node, ABC):
    _fdfg: FDFG

    def __init__(self, fdfg: FDFG) -> None:
        super().__init__()
        self._fdfg = fdfg

    @property
    def fdfg(self) -> FDFG:
        return self._fdfg


class LoopNode(ParametricNode):
    _index_symbol_names: set[Identifier]

    def __init__(self, index_symbol_names: set[Identifier], fdfg: FDFG) -> None:
        super().__init__(fdfg)
        self._index_symbol_names = index_symbol_names

    @property
    def index_symbol_names(self) -> set[Identifier]:
        return self._index_symbol_names


class ReductionNode(ParametricNode):
    _symbol_name: Identifier
    _index_symbol_names: set[Identifier]

    def __init__(
        self, symbol_name: Identifier, index_symbol_names: set[Identifier], fdfg: FDFG
    ) -> None:
        super().__init__(fdfg)
        self._symbol_name = symbol_name
        self._index_symbol_names = index_symbol_names

    @property
    def symbol_name(self) -> Identifier:
        return self._symbol_name

    @property
    def index_symbol_names(self) -> set[Identifier]:
        return self._index_symbol_names
