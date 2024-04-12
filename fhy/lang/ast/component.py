# TODO Jason: Add docstring
from typing import List
from .base import Component
from .expression import Identifier
from .qualified_type import QualifiedType
from .statement import Statement


class Argument(object):
    """Function Argument Container Node

    Args:
        name (Identifier): Argument Name

    """
    _name: Identifier
    _qualified_type: QualifiedType

    def __init__(self,
                 name: Identifier,
                 qualified_type: QualifiedType,
                 ) -> None:
        self._name = name
        self._qualified_type = qualified_type

    @property
    def name(self) -> Identifier:
        return self._name

    @property
    def qualified_type(self) -> QualifiedType:
        return self._qualified_type

    def visit_attrs(self) -> List[str]:
        return ["_name", "_qualified_type"]

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

    @property
    def args(self) -> List[Argument]:
        return self._args

    @property
    def body(self) -> List[Statement]:
        return self._body

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
