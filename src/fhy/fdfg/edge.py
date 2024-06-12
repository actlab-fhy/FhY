from dataclasses import dataclass
from fhy.ir.identifier import Identifier


@dataclass(frozen=True)
class Edge(object):
    symbol_name: Identifier
    source_arg_index: int
    destination_arg_index: int
