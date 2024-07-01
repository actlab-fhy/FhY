"""Fractalized (i.e., with a sub-f-DFG for each node) nodes for the f-DFG."""
from abc import ABC
from .base import Node
from fhy.ir.identifier import Identifier
from ..core import FDFG


class FractalizedNode(Node, ABC):
    """Base abstract fractalized f-DFG node.

    Args:
        fdfg (FDFG | None): Sub-f-DFG for the node. Defaults to None.

    """
    _fdfg: FDFG | None

    def __init__(self, fdfg: FDFG | None = None) -> None:
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
    """f-DFG node representing a function call.

    Args:
        symbol_name (Identifier): Symbol name of the function.
        fdfg (FDFG | None): f-DFG for the function. Defaults to None.

    """
    _symbol_name: Identifier

    def __init__(self, symbol_name: Identifier, fdfg: FDFG | None = None) -> None:
        super().__init__(fdfg=fdfg)
        self._symbol_name = symbol_name

    @property
    def symbol_name(self) -> Identifier:
        return self._symbol_name
