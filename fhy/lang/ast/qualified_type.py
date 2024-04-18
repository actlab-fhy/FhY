""" """
from dataclasses import dataclass
from typing import List, Optional

from fhy.ir import Type, TypeQualifier

from .core import ASTNode


@dataclass(frozen=True, kw_only=True)
class QualifiedType(ASTNode):
    """Qualified Type Container

    Args:
        _base_type (Type): Primitive or Generic Type (e.g. float, int, T)
        _type_qualifier (Optional[TypeQualifier]): Qualifying Type Identifier

    """

    base_type: Optional[Type]
    type_qualifier: Optional[TypeQualifier]

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_base_type", "_type_qualifier"])
        return attrs
