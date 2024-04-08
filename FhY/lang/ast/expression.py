# TODO Jason: Add docstring
from abc import ABC
from enum import StrEnum
from typing import List
from .base import Expression, Identifier
from .type import Type


class UnaryOperation(StrEnum):
    # TODO Jason: Add docstring
    NEGATIVE = "-"
    BITWISE_NOT = "~"
    LOGICAL_NOT = "!"


class UnaryExpression(Expression):
    # TODO Jason: Add docstring
    _operation: UnaryOperation
    _expression: Expression

    # TODO Jason: Implement the functionality of this class


class BinaryOperation(StrEnum):
    # TODO Jason: Add docstring
    MULTIPLICATION = "*"
    DIVISION = "/"
    ADDITION = "+"
    SUBTRACTION = "-"
    LEFT_SHIT = "<<"
    RIGHT_SHIFT = ">>"
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    GREATER_THAN = ">"
    GREATHER_THAN_OR_EQUAL = ">="
    EQUAL_TO = "=="
    NOT_EQUAL_TO = "!="
    BITWISE_AND = "&"
    BITWISE_XOR = "^"
    BITWISE_OR = "|"
    LOGICAL_AND = "&&"
    LOGICAL_OR = "||"


class BinaryExpression(Expression):
    # TODO Jason: Add docstring
    _operation: BinaryOperation
    _left_expression: Expression
    _right_expression: Expression

    # TODO Jason: Implement the functionality of this class


class TernaryExpression(Expression):
    # TODO Jason: Add docstring
    _predicate: Expression
    _true_expression: Expression
    _false_expression: Expression

    # TODO Jason: Implement the functionality of this class


class TupleAccessExpression(Expression):
    # TODO Jason: Add docstring
    _expression: Expression
    _element_index: int

    # TODO Jason: Implement the functionality of this class


class FunctionExpression(Expression):
    # TODO Jason: Add docstring
    _template_types: List[Type]
    _indices: List[Identifier]
    _args: List[Expression]

    # TODO Jason: Implement the functionality of this class


class TensorAccessExpression(Expression):
    # TODO Jason: Add docstring
    _expression: Expression
    _indices: List[Expression]

    # TODO Jason: Implement the functionality of this class


class TupleExpression(Expression):
    # TODO Jason: Add docstring
    _expressions: List[Expression]

    # TODO Jason: Implement the functionality of this class


class IdentifierExpression(Expression):
    # TODO Jason: Add docstring
    _identifier: Identifier

    # TODO Jason: Implement the functionality of this class


class Literal(Expression, ABC):
    # TODO Jason: Add docstring
    ...


class IntLiteral(Literal):
    # TODO Jason: Add docstring
    _value: int

    def __init__(self, value: int):
        self._value = value

    @property
    def value(self) -> int:
        # TODO Jason: Add docstring
        return self._value


class FloatLiteral(Literal):
    # TODO Jason: Add docstring
    _value: float

    def __init__(self, value: float):
        self._value = value

    @property
    def value(self) -> float:
        # TODO Jason: Add docstring
        return self._value
