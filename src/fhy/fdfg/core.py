"""f-DFG."""
from abc import ABC
from typing import Any
import networkx as nx
from typing import Optional
from fhy.ir.identifier import Identifier
from fhy.ir.type import Type
from .node.base import Node
from .node.io import SourceNode, SinkNode
from .edge import Edge


# TODO: need to add an ordering for the input and output nodes for calls
class FDFG(object):
    _graph: nx.MultiDiGraph
    _source_node_name: Identifier
    _sink_node_name: Identifier

    def __init__(self, graph: Optional[nx.MultiDiGraph] = None) -> None:
        if graph is not None:
            self._set_graph(graph)
        else:
            self._graph = nx.MultiDiGraph()
            self._source_node_name = Identifier("source")
            self._sink_node_name = Identifier("sink")
            self._graph.add_node(self._source_node_name, data=SourceNode())
            self._graph.add_node(self._sink_node_name, data=SinkNode())

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    @graph.setter
    def graph(self, graph: nx.MultiDiGraph) -> None:
        self._set_graph(graph)

    def _set_graph(self, graph: nx.MultiDiGraph) -> None:
        source_node_names: list[Identifier] = [node_name for node_name, node_attr in graph.nodes(data=True) if isinstance(node_attr["data"], SourceNode)]
        sink_node_names: list[Identifier] = [node_name for node_name, node_attr in graph.nodes(data=True) if isinstance(node_attr["data"], SinkNode)]

        if len(source_node_names) != 1:
            raise ValueError("Expected exactly one source node in the graph.")
        if len(sink_node_names) != 1:
            raise ValueError("Expected exactly one sink node in the graph.")

        self._source_node_name = source_node_names[0]
        self._sink_node_name = sink_node_names[0]
        self._graph = graph

    @property
    def source_node_name(self) -> Identifier:
        return self._source_node_name

    @property
    def sink_node_name(self) -> Identifier:
        return self._sink_node_name

    def get_number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()

    def add_node(self, node_name: Identifier, node: Node) -> None:
        if isinstance(node, SourceNode) or isinstance(node, SinkNode):
            raise ValueError("Cannot add a source or sink node to an f-DFG.")
        self._graph.add_node(node_name, data=node)

    def remove_node(self, node_name: Identifier) -> None:
        if node_name == self._source_node_name or node_name == self._sink_node_name:
            raise ValueError("Cannot remove the source or sink node from an f-DFG.")
        self._graph.remove_node(node_name)

    def add_edge(
        self,
        source_node_name: Identifier,
        sink_node_name: Identifier,
        source_arg_index: int,
        destination_arg_index: int,
        symbol_name: Identifier | None = None,
    ) -> None:
        self._graph.add_edge(source_node_name, sink_node_name, data=Edge(symbol_name, source_arg_index, destination_arg_index))

    def remove_edge(
        self, source_node_name: Identifier, sink_node_name: Identifier
    ) -> None:
        self._graph.remove_edge(source_node_name, sink_node_name)

    def get_predecessors(self, node_name: Identifier) -> set[Identifier]:
        return set(self._graph.predecessors(node_name))

    def get_successors(self, node_name: Identifier) -> set[Identifier]:
        return set(self._graph.successors(node_name))

    def get_input_symbol_names(self) -> list[Identifier]:
        edges_from_source_node: list[tuple[Identifier, Identifier, Edge]] = [(u, v, d["data"]) for u, v, d in self._graph.edges(self._source_node_name, data=True)]
        return [x[2].symbol_name for x in sorted(edges_from_source_node, key=lambda x: x[2].passed_arg_index)]

    def get_output_symbol_names(self) -> list[Identifier]:
        edges_to_sink_node: list[tuple[Identifier, Identifier, Edge]] = [(u, v, d["data"]) for u, v, d in self._graph.edges(data=True) if v == self._sink_node_name]
        return [x[2].symbol_name for x in sorted(edges_to_sink_node, key=lambda x: x[2].passed_arg_index)]
