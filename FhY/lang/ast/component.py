# TODO Jason: Add docstring
from abc import ABC
from typing import Any, List, Optional
from .base import Component
from .expression import Identifier
from .base import ASTNode
from .statement import Statement
from .type import Type, TypeQualifier


class QualifiedType(ASTNode):
    """Qualified Type Container

    Args:
        _type (Type): Primitive or Generic Type (e.g. float, int, T)
        _type_qualifier (Optional[TypeQualifier]): Qualifying Type, (i.e. input, output, param, state)

    e.g. Return Types are not assigned a proper name

    """
    def __init__(self,
                 _type: Type,
                 _type_qualifier: Optional[TypeQualifier] = None
                 ) -> None:
        super().__init__()
        self._type = _type
        self._type_qualifier = _type_qualifier

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_type", "_type_qualifier"])
        return attrs


class Argument(QualifiedType):
    """Function Argument Container Node

    Args:
        _name (Identifier): Argument Name
        _type (Type): Primitive or Generic Type (e.g. float, int, T)
        _type_qualifier (Optional[TypeQualifier]): Qualifying Type, (i.e. input, output, param, state)

    """
    def __init__(self,
                 _name: Identifier,
                 _type: Type,
                 _type_qualifier: Optional[TypeQualifier] = None,
                 ) -> None:
        super().__init__(_type, _type_qualifier)
        self._name = _name

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_name"])
        return attrs

    # TODO Jason: Implement the functionality of this class


class Procedure(Component):
    """Fhy Procedure Node"""

    def __init__(self,
                 name: Identifier,
                 _args: List[Argument],
                 _body: List[Statement]
                 ) -> None:
        super().__init__(name)
        self._args = _args
        self._body = _body

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_args", "_body"])
        return attrs


class Operation(Component):
    """Operation

    Args:
        name (Identifier): variable Name
        _args (List[Argument]): list of Arguments
        _body (List[Statement]): list of Statements, defining the body of the function
        _ret_type (QualifiedType): Type information of the Returned Value

    """
    def __init__(self, 
                 name: Identifier,
                 _args: List[Argument],
                 _body: List[Statement],
                 _ret_type: QualifiedType
                 ) -> None:
        super().__init__(name)
        self._args = _args
        self._body = _body
        self._ret_type = _ret_type

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_args", "_body", "_ret_type"])
        return attrs


class Native(Component):
    def __init__(self, name: Identifier, _args: List[Argument]) -> None:
        super().__init__(name)
        self._args = _args

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_args"])
        return attrs
