"""Primitive nodes for the f-DFG."""
from typing import Any
from .base import Node
from ..op import Op


class PrimitiveNode(Node):
    """Primitive node for the f-DFG.

    Args:
        op (Op): Operation of the node.

    """
    _op: Op

    def __init__(self, op: Op) -> None:
        super().__init__()
        self._op = op

    @property
    def op(self) -> Op:
        return self._op


class LiteralNode(Node):
    """Literal node for the f-DFG.

    Args:
        value (Any): Value of the node.

    """
    _value: Any

    def __init__(self, value: Any) -> None:
        super().__init__()
        self._value = value

    @property
    def value(self) -> Any:
        return self._value
