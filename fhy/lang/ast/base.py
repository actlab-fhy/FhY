from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from ..span import Span


@dataclass(frozen=True, kw_only=True)
class ASTNode(ABC):
    """Core abstract AST node."""

    span: Optional[Span] = field(default=None)

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
