"""Qualified type AST node."""

from dataclasses import dataclass
from typing import List

from fhy.ir.type import Type as IRType
from fhy.ir.type import TypeQualifier as IRTypeQualifier

from .base import ASTNode


@dataclass(frozen=True, kw_only=True)
class QualifiedType(ASTNode):
    """Qualified type AST node.

    Args:
        base_type (IRType): Type of the qualified type.
        type_qualifier (IRTypeQualifier): Qualifier of the type.

    Attributes:
        base_type (IRType): Type of the qualified type.
        type_qualifier (IRTypeQualifier): Qualifier of the type.

    """

    base_type: IRType
    type_qualifier: IRTypeQualifier

    def get_visit_attrs(self) -> List[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["base_type", "type_qualifier"])
        return attrs
