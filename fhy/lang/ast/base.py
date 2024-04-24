"""Base Abstract AST Node Definition.

Typical Usage:

    .. code-block:: python

        from dataclasses import dataclass
        from fhy.lang.ast.base import ASTNode

        @dataclass(frozen=True, kw_only=True)
        class NewASTNode(ASTNode):
            body: List[ASTNode] = field(default_factory=list)

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from ..span import Span


@dataclass(frozen=True, kw_only=True)
class ASTNode(ABC):
    """Core abstract AST node.

    Args:
        span (Optional[Span]): Define Position in source Code

    """

    span: Optional[Span] = field(default=None)

    @classmethod
    def keyname(cls) -> str:
        """Class Node Name"""
        if hasattr(cls, "__qualname__"):
            return cls.__qualname__
        return cls.__name__

    @abstractmethod
    def visit_attrs(self) -> List[str]:
        """Returns a list of relevant node fields"""
        return ["span"]
