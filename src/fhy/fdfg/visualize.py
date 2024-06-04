import networkx as nx
from pathlib import Path
from .core import FDFG, Node, Op, IONode, FractalizedNode, PrimitiveNode, Edge


def export_fdfg_to_dot(fdfg: FDFG, dot_file_path: Path) -> None:
    raise NotImplementedError("export_fdfg_to_dot")


def plot_fdfg(fdfg: FDFG) -> None:
    import matplotlib.pyplot as plt

    graph = fdfg._graph
    nodes = graph.nodes(data=True)
    edges = graph.edges()

    node_labels = {}
    for node_name, node in nodes:
        if isinstance(node, IONode):
            node_labels[node] = node.symbol_name.name_hint
        elif isinstance(node, FractalizedNode):
            node_labels[node] = node_name.name_hint
        elif isinstance(node, PrimitiveNode):
            node_labels[node] = node.op.name.name_hint
    # edge_labels = {(u, v): edge for u, v, edge in edges}

    node_positions = nx.spring_layout(graph)
    nx.draw_networkx(graph, node_positions, node_size=700, node_color="skyblue",
                     with_labels=True, labels=node_labels)
    nx.draw_networkx_edge_labels(graph, node_positions)
    plt.savefig("fdfg.png")
