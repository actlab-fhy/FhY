# TODO Jason: Add docstring
from abc import ABC
from typing import List, Optional
from .expression import Identifier
from .base import Statement
from .expression import Expression
from .type import Type, TypeQualifier


class DeclarationStatement(Statement):
    # TODO Jason: Add docstring
    _variable_name: Identifier
    _variable_type: Type
    _variable_type_qual: TypeQualifier
    _expression: Optional[Expression] = None

    # TODO Jason: Implement the functionality of this class


class ExpressionStatement(Statement):
    # TODO Jason: Add docstring
    _left: Optional[Identifier] = None
    _index: List[Expression] = []
    _right: Expression

    # TODO Jason: Implement the functionality of this class


class ForAllStatement(Statement):
    # TODO Jason: Add docstring
    _index: Expression
    _body: List[Statement]

    # TODO Jason: Implement the functionality of this class


class BranchStatement(Statement):
    # TODO Jason: Add docstring
    _predicate: Expression
    _true_body: List[Statement]
    _false_body: List[Statement]

    # TODO Jason: Implement the functionality of this class
