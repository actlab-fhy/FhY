"""All Nodes defined within this Module are subclasses of core.Component ASTNode

Component ASTNodes:
    Argument: Argument identifier node
    Procedure: FhY procedure function node
    Operation: FhY operation Function node
    Native: (Not yet supported)

"""

from dataclasses import dataclass, field
from typing import List

from .core import ASTNode, Function, Component
from .expression import Identifier
from .qualified_type import QualifiedType
from .statement import Statement


@dataclass(frozen=True, kw_only=True)
class Import(Component):
    """Import ASTNode
    """
    name: Identifier

    def visit_attrs(self) -> List[str]:
        return ["name"]


@dataclass(frozen=True, kw_only=True)
class Argument(ASTNode):
    """Function Argument ASTNode.

    Args:
        name (Identifier): Variable Identifier of the argument
        qualified_type (QualifiedType): Qualified Type of given argument

    """

    name: Identifier
    qualified_type: QualifiedType

    def visit_attrs(self) -> List[str]:
        return ["name", "qualified_type"]


@dataclass(frozen=True, kw_only=True)
class Procedure(Function):
    """FhY Procedure function ASTNode.

    Args:
        args (List[Argument]): list of Arguments
        body (List[Statement]): list of Statements, defining the body of the procedure

    """

    args: List[Argument] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        attrs: List[str] = super().visit_attrs()
        attrs.extend(["args", "body"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class Operation(Function):
    """FhY Operation Function ASTNode

    Args:
        args (List[Argument]): list of Arguments
        body (List[Statement]): list of Statements, defining the body of the operation
        ret_type (QualifiedType): Type information of the Returned Value

    """

    args: List[Argument] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)
    return_type: QualifiedType

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["args", "body", "return_type"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class Native(Function):
    """FhY Native Function ASTNode"""

    args: List[Argument] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["args"])
        return attrs
