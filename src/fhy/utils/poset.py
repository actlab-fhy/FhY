from typing import Generic, TypeVar
import networkx as nx

T = TypeVar("T")


class POSet(Generic[T]):
    """A partially ordered set (poset)."""
    _graph: nx.DiGraph

    def __init__(self):
        self._graph = nx.DiGraph()

    def __iter__(self):
        return iter(nx.topological_sort(self._graph))

    def __len__(self) -> int:
        return len(self._graph.nodes)

    def add_element(self, element: T) -> None:
        """Add an element to the poset."""
        self._graph.add_node(element)

    def add_order(self, lower: T, upper: T) -> None:
        """Add an order relation between two elements."""
        if not self._graph.has_node(lower) or not self._graph.has_node(upper):
            raise RuntimeError("Expected nodes to be added to the poset before adding an order.")
        if nx.has_path(self._graph, upper, lower):
            raise RuntimeError(f"Expected {lower} to be less than {upper} in the poset, but {upper} is already less than {lower}.")
        self._graph.add_edge(lower, upper)

    def is_less_than(self, lower: T, upper: T) -> bool:
        """Check if one element is less than another."""
        return nx.has_path(self._graph, lower, upper)

    def is_greater_than(self, lower: T, upper: T) -> bool:
        """Check if one element is greater than another."""
        return nx.has_path(self._graph, upper, lower)
