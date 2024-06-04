from abc import ABC
from typing import Any
import networkx as nx
from typing import Optional
from fhy.ir.identifier import Identifier
from fhy.ir.type import Type



class Node(ABC):
    ...





class FDFG(object):
    _graph: nx.MultiDiGraph

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()





class Op(object):
    _name: Identifier
    # _input_signature: list[Type]
    # _output_signature: list[Type]
    # _template_types: list[Type]

    def __init__(
        self,
        name: Identifier,
        # input_signature: list[Type],
        # output_signature: list[Type],
        # template_types: Optional[list[Type]] = None,
    ) -> None:
        self._name = name
        # self._input_signature = input_signature
        # self._output_signature = output_signature
        # self._template_types = template_types or []

    @property
    def name(self) -> Identifier:
        return self._name

    # @property
    # def input_signature(self) -> list[Type]:
    #     return self._input_signature

    # @property
    # def output_signature(self) -> list[Type]:
    #     return self._output_signature

    # @property
    # def template_types(self) -> list[Type]:
    #     return self._template_types



class IONode(Node, ABC):
    _symbol_name: Identifier

    def __init__(self, symbol_name: Identifier) -> None:
        super().__init__()
        self._symbol_name = symbol_name

    @property
    def symbol_name(self) -> Identifier:
        return self._symbol_name


class SourceNode(IONode):
    pass


class SinkNode(IONode):
    pass


class FractalizedNode(Node):
    _input_node_names: list[Identifier]
    _output_node_names: list[Identifier]
    _sub_graph: FDFG

    def __init__(
        self,
        input_node_names: list[Identifier],
        output_node_names: list[Identifier],
        sub_graph: FDFG,
    ) -> None:
        super().__init__()
        self._input_node_names = input_node_names
        self._output_node_names = output_node_names
        self._sub_graph = sub_graph


class ForLoopNode(FractalizedNode):
    _index_symbol_name: Identifier


class PrimitiveNode(Node):
    _op: Op

    def __init__(self, op: Op) -> None:
        super().__init__()
        self._op = op

    @property
    def op(self) -> Op:
        return self._op


class Edge(object):
    _name: Identifier
    _passed_arg_index: int
