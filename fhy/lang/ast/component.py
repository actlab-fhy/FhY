# TODO Jason: Add docstring
from dataclasses import dataclass, field
from typing import List, Optional

from .base import ASTNode, Function
from .expression import Identifier
from .qualified_type import QualifiedType
from .statement import Statement


@dataclass(frozen=True, kw_only=True)
class Argument(ASTNode):
    # TODO Jason: Add docstring
    name: Optional[Identifier] = field(default=None)
    qualified_type: Optional[QualifiedType] = field(default=None)

    def visit_attrs(self) -> List[str]:
        return ["name", "qualified_type"]


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


class Operation(Function):
    """Operation

    Args:
        name (Identifier): variable Name
        _args (List[Argument]): list of Arguments
        _body (List[Statement]): list of Statements, defining the body of the function
        _ret_type (QualifiedType): Type information of the Returned Value

    """

    def __init__(
        self,
        name: Identifier,
        _args: List[Argument],
        _body: List[Statement],
        _ret_type: QualifiedType,
    ) -> None:
        super().__init__(name)  # type: ignore[misc]
        self._args = _args
        self._body = _body
        self._ret_type = _ret_type

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_args", "_body", "_ret_type"])
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
