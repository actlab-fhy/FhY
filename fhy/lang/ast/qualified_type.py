from typing import List, Optional

from fhy.ir import Type, TypeQualifier

from .base import ASTNode


class QualifiedType(ASTNode):
    """Qualified Type Container

    Args:
        _base_type (Type): Primitive or Generic Type (e.g. float, int, T)
        _type_qualifier (Optional[TypeQualifier]): Qualifying Type Identifier

    """

    _base_type: Optional[Type]
    _type_qualifier: Optional[TypeQualifier]

    def __init__(
        self,
        base_type: Optional[Type] = None,
        type_qualifier: Optional[TypeQualifier] = None,
    ) -> None:
        super().__init__()
        self._base_type = base_type
        self._type_qualifier = type_qualifier

    @property
    def base_type(self) -> Optional[Type]:
        return self._base_type

    @property
    def type_qualifier(self) -> Optional[TypeQualifier]:
        return self._type_qualifier

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_base_type", "_type_qualifier"])
        return attrs
