# TODO Jason: Add docstring
from abc import ABC
from typing import List, Optional
from .base import Component
from .expression import Identifier
from .base import ASTNode
from .statement import Statement
from .type import Type, TypeQualifier


class Argument(ASTNode):
    # TODO Jason: Add docstring
    _name: Identifier
    _type: Type
    _type_qualifier: Optional[TypeQualifier]

    # TODO Jason: Implement the functionality of this class


class Procedure(Component):
    _args: List[Argument]
    _body: List[Statement]

    # TODO Jason: Implement the functionality of this class


class Operation(Component):
    _args: List[Argument]
    _body: List[Statement]

    # TODO Jason: Implement the functionality of this class


class Native(Component):
    _args: List[Argument]

    # TODO Jason: Implement the functionality of this class
