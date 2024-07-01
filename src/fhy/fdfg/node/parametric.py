"""Parametric nodes for the f-DFG."""
from abc import ABC
from ..node import Node
from fhy.ir.identifier import Identifier
from ..core import FDFG
from fhy.ir.expression import Expression


class ParametricNode(Node, ABC):
    """Base abstract parametric f-DFG node.

    Args:
        fdfg (FDFG): The parameter sub-f-DFG for the node.

    """
    _fdfg: FDFG

    def __init__(self, fdfg: FDFG) -> None:
        super().__init__()
        self._fdfg = fdfg

    @property
    def fdfg(self) -> FDFG:
        return self._fdfg


class LoopNode(ParametricNode):
    """f-DFG node representing a loop.

    Args:
        index_symbol_names (set[Identifier]): Symbol names of the loop indices
            the loop node iterates over.
        fdfg (FDFG): f-DFG for the body of the loop.

    """
    _index_symbol_names: set[Identifier]
    # TODO: name this attribute better
    # The key is the ename of the edge from the source node or to the sink node
    # The value is a list of expressions describing the index expression used
    # to access the array; the indices would be applied starting at the outermost
    # dimension and working inwards
    _symbol_edge_name_access_indices: dict[Identifier, list[Expression]]

    def __init__(self, index_symbol_names: set[Identifier], symbol_edge_name_access_indices: dict[Identifier, list[Expression]], fdfg: FDFG) -> None:
        super().__init__(fdfg)
        self._index_symbol_names = index_symbol_names
        self._symbol_edge_name_access_indices = symbol_edge_name_access_indices

    @property
    def index_symbol_names(self) -> set[Identifier]:
        return self._index_symbol_names


class ReductionNode(LoopNode):
    """f-DFG node representing a reduction.

    Args:
        symbol_name (Identifier): Symbol name of the reduction operation.
        reduced_index_symbol_names (set[Identifier]): Symbol names of the loop
            indices the reduction node reduces.
        index_symbol_names (set[Identifier]): Symbol names of the loop indices
            the reduction node iterates over.
        fdfg (FDFG): f-DFG for the body of the reduction.

    """
    _symbol_name: Identifier
    _reduced_index_symbol_names: set[Identifier]

    def __init__(
        self,
        symbol_name: Identifier,
        reduced_index_symbol_names: set[Identifier],
        index_symbol_names: set[Identifier],
        symbol_edge_name_access_indices: dict[Identifier, list[Expression]],
        fdfg: FDFG
    ) -> None:
        super().__init__(index_symbol_names, symbol_edge_name_access_indices, fdfg)
        self._symbol_name = symbol_name
        self._reduced_index_symbol_names = reduced_index_symbol_names

    @property
    def symbol_name(self) -> Identifier:
        return self._symbol_name

    @property
    def reduced_index_symbol_names(self) -> set[Identifier]:
        return self._reduced_index_symbol_names
