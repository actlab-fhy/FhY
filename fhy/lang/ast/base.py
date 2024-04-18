from abc import ABC, abstractmethod
from typing import List
from ..span import Span


class ASTNode(ABC):
    """Core abstract AST node."""

    _span: Span

    # TODO: is this method necessary?
    @classmethod
    def keyname(cls) -> str:
        """Class Node Name"""
        if hasattr(cls, "__qualname__"):
            return cls.__qualname__
        return cls.__name__

    @abstractmethod
    def visit_attrs(self) -> List[str]:
        # TODO Jason: Add docstring
        return []
