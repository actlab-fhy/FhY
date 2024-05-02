"""Defines a Qualified Type ASTNode."""

from dataclasses import dataclass
from typing import List

from fhy import ir

from .core import ASTNode


@dataclass(frozen=True, kw_only=True)
class QualifiedType(ASTNode):
    """Qualified Type ASTNode defining both primitive and qualified types.

    Args:
        base_type (Type): Primitive or Generic Type
        type_qualifier (TypeQualifier): Qualifying Type Identifier

    """

    base_type: ir.Type
    type_qualifier: ir.TypeQualifier

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["base_type", "type_qualifier"])
        return attrs
