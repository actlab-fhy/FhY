# TODO Jason: Add docstring
from dataclasses import dataclass, field
from typing import List, Optional

from .core import ASTNode, Function
from .directory import register_ast_node
from .expression import Identifier
from .qualified_type import QualifiedType
from .statement import Statement


@register_ast_node
@dataclass(frozen=True, kw_only=True)
class Argument(ASTNode):
    # TODO Jason: Add docstring
    name: Optional[Identifier] = field(default=None)
    qualified_type: Optional[QualifiedType] = field(default=None)

    def visit_attrs(self) -> List[str]:
        return ["name", "qualified_type"]


@register_ast_node
@dataclass(frozen=True, kw_only=True)
class Procedure(Function):
    """FhY procedure node"""

    args: List[Argument] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs: List[str] = super().visit_attrs()
        attrs.extend(["args", "body"])
        return attrs


@register_ast_node
@dataclass(frozen=True, kw_only=True)
class Operation(Function):
    """FhY Operation Node

    Args:
        name (Identifier): variable Name
        args (List[Argument]): list of Arguments
        body (List[Statement]): list of Statements, defining the body of the function
        ret_type (QualifiedType): Type information of the Returned Value

    """

    args: List[Argument] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)
    ret_type: QualifiedType

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["args", "body", "ret_type"])
        return attrs


class Native(Function):
    def __init__(self, name: Identifier, _args: List[Argument]) -> None:
        super().__init__(name)  # type: ignore[misc]
        self._args = _args

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_args"])
        return attrs
