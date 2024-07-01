"""Fractalized Data-Flow Graph (f-DFG)."""
from .node import (
    Node,
    SourceNode,
    SinkNode,
    FunctionNode,
    ReductionNode,
    LoopNode,
    PrimitiveNode,
    LiteralNode,
)
from .op import Op
from .edge import Edge
from .core import FDFG
