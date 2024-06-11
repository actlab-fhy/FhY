from abc import ABC
from typing import Any
import networkx as nx
from typing import Optional
from fhy.ir.identifier import Identifier
from fhy.ir.type import Type
from .node.base import Node
from .node.io import SourceNode, SinkNode


# TODO: need to add an ordering for the input and output nodes for calls
class FDFG(object):
    _graph: nx.MultiDiGraph

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    @graph.setter
    def graph(self, graph: nx.MultiDiGraph) -> None:
        self._graph = graph

    # def merge_fdfg(self, fdfg: "FDFG") -> None:
    #     self._graph = nx.compose(self._graph, fdfg.graph)

    def add_node(self, node_name: Identifier, node: Node) -> None:
        self._graph.add_node(node_name, data=node)

    def remove_node(self, node_name: Identifier) -> None:
        self._graph.remove_node(node_name)

    def add_edge(
        self, source_node_name: Identifier, sink_node_name: Identifier
    ) -> None:
        self._graph.add_edge(source_node_name, sink_node_name)

    def remove_edge(
        self, source_node_name: Identifier, sink_node_name: Identifier
    ) -> None:
        self._graph.remove_edge(source_node_name, sink_node_name)

    def predecessors(self, node_name: Identifier) -> list[Identifier]:
        return list(self._graph.predecessors(node_name))

    def successors(self, node_name: Identifier) -> list[Identifier]:
        return list(self._graph.successors(node_name))

    def get_input_node_names(self) -> list[Identifier]:
        return [tup[0] for tup in self._get_nodes_and_data(SourceNode)]

    def get_input_nodes(self) -> list["SourceNode"]:
        return [tup[1] for tup in self._get_nodes_and_data(SourceNode)]

    def get_output_node_names(self) -> list[Identifier]:
        return [tup[0] for tup in self._get_nodes_and_data(SinkNode)]

    def get_output_nodes(self) -> list["SinkNode"]:
        return [tup[1] for tup in self._get_nodes_and_data(SinkNode)]

    def _get_nodes_and_data(
        self, node_type: type[Node]
    ) -> list[tuple[Identifier, Node]]:
        nodes_and_attrs = filter(
            lambda tup: isinstance(tup[1]["data"], node_type),
            self._graph.nodes(data=True),
        )
        return [(tup[0], tup[1]["data"]) for tup in nodes_and_attrs]
