"""Input/Output nodes for the f-DFG."""
from abc import ABC
from .base import Node


class IONode(Node, ABC):
    """Base abstract I/O node for the f-DFG."""


class SourceNode(IONode):
    """Source node for the f-DFG."""


class SinkNode(IONode):
    """Sink node for the f-DFG."""
