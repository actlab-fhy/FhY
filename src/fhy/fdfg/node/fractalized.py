from abc import ABC
from typing import Optional
from .base import Node
from fhy.ir.identifier import Identifier
from ..core import FDFG


class FractalizedNode(Node, ABC):
    _fdfg: Optional[FDFG]

    def __init__(self, fdfg: Optional[FDFG] = None) -> None:
        super().__init__()
        self._fdfg = fdfg

    @property
    def fdfg(self) -> FDFG:
        if self.is_fdfg_set() is False:
            raise RuntimeError(f"f-DFG for node {self} is not set.")
        return self._fdfg

    @fdfg.setter
    def fdfg(self, fdfg: FDFG) -> None:
        self._fdfg = fdfg

    def is_fdfg_set(self) -> bool:
        return self._fdfg is not None


class FunctionNode(FractalizedNode):
    _symbol_name: Identifier

    def __init__(self, symbol_name: Identifier, fdfg: Optional[FDFG] = None) -> None:
        super().__init__(fdfg=fdfg)
        self._symbol_name = symbol_name

    @property
    def symbol_name(self) -> Identifier:
        return self._symbol_name
