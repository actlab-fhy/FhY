"""Base abstract AST node.

Typical Usage:

    .. code-block:: python

        from fhy.lang.ast.base import ASTNode

        class NewASTNode(ASTNode):
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from ..span import Span


@dataclass(frozen=True, kw_only=True)
class ASTNode(ABC):
    """A node in the FhY AST.

    Attributes:
        span (Span): The Span of the node in the source code.

    """

    # TODO: Make span optional? A default span of (0,0,0,0) doesn't make sense
    span: Span = field(default_factory=lambda: Span(0, 0, 0, 0))

    @classmethod
    def get_key_name(cls) -> str:
        """Return the unique name of the node."""
        return cls.__name__

    @abstractmethod
    def get_visit_attrs(self) -> List[str]:
        """Return a list of node fields defining the contents of the node."""
        return ["span"]
