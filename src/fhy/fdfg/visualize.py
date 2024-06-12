import networkx as nx
from pathlib import Path
from fhy.ir.identifier import Identifier
from .core import FDFG, Node
from .node.io import SourceNode, SinkNode, IONode
from .node.fractalized import FractalizedNode, FunctionNode
from .node.parametric import LoopNode
from .node.io import SinkNode, SourceNode
from .node.primitive import PrimitiveNode, LiteralNode
from collections import deque


def export_fdfg_to_dot(fdfg: FDFG, dot_file_path: Path) -> None:
    raise NotImplementedError("export_fdfg_to_dot")


def _plot_fdfg_granularity(fdfg: FDFG, output_file_path: Path) -> None:
    import matplotlib.pyplot as plt

    graph = fdfg._graph
    nodes = graph.nodes(data=True)
    edges = graph.edges()

    node_labels: dict[Identifier, str] = {}
    node_colors: list[str] = []
    for node_name, node_attr in nodes:
        if "data" not in node_attr:
            node_labels[node_name] = "error"
            node_colors.append("red")
            continue

        node = node_attr["data"]
        if isinstance(node, IONode):
            node_labels[node_name] = node.symbol_name.name_hint
            if isinstance(node, SourceNode):
                node_colors.append("lime")
            elif isinstance(node, SinkNode):
                node_colors.append("coral")
            else:
                raise RuntimeError(f"Unexpected IONode type: {type(node)}")
        elif isinstance(node, FunctionNode):
            node_labels[node_name] = node_name.name_hint
            node_colors.append("skyblue")
        elif isinstance(node, PrimitiveNode):
            node_labels[node_name] = node.op.name.name_hint
            node_colors.append("aqua")
        else:
            node_labels[node_name] = node_name.name_hint
            node_colors.append("wheat")
    # edge_labels = {(u, v): edge for u, v, edge in edges}

    node_positions = nx.spring_layout(graph)
    nx.draw_networkx(
        graph,
        node_positions,
        node_size=700,
        node_color=node_colors,
        with_labels=True,
        labels=node_labels,
    )
    # nx.draw_networkx_edge_labels(graph, node_positions)
    plt.savefig(output_file_path)
    plt.close()


def plot_fdfg(fdfg: FDFG) -> None:
    fdfgs: deque[tuple[Identifier, FDFG]] = deque()
    fdfgs.append((Identifier("root"), fdfg))
    while fdfgs:
        name, fdfg = fdfgs.popleft()
        _plot_fdfg_granularity(fdfg, Path(f"fdfg_{name.name_hint}.png"))

        for node_name, node_attrs in fdfg.graph.nodes(data=True):
            if "data" not in node_attrs:
                continue
            node = node_attrs["data"]
            if isinstance(node, FractalizedNode) and node.is_fdfg_set():
                fdfgs.append((node_name, node.fdfg))
