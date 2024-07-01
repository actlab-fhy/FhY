"""f-DFG edge."""
from dataclasses import dataclass
from fhy.ir.identifier import Identifier


@dataclass(frozen=True)
class Edge(object):
    """f-DFG edge.

    Args:
        symbol_name (Identifier | None): Symbol name of the variable associated
            with the edge. If None, the edge is an anonymous edge.
        source_arg_index (int): ...
        destination_arg_index (int): ...
    """
    symbol_name: Identifier | None
    source_arg_index: int
    destination_arg_index: int
