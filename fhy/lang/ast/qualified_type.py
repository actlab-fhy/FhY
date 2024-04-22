""" """

from dataclasses import dataclass
from typing import List, Optional

from fhy import ir

from .core import ASTNode
from .directory import register_ast_node


@register_ast_node
@dataclass(frozen=True, kw_only=True)
class QualifiedType(ASTNode):
    """Qualified Type Container

    Args:
        base_type (Type): Primitive or Generic Type (e.g. float, int, T)
        type_qualifier (Optional[TypeQualifier]): Qualifying Type Identifier

    """

    base_type: ir.Type
    type_qualifier: ir.TypeQualifier

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["base_type", "type_qualifier"])
        return attrs
