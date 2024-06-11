from abc import ABC
from .base import Node


class IONode(Node, ABC): ...


class SourceNode(IONode): ...


class SinkNode(IONode): ...
