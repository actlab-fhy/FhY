# TODO Jason: Add docstring
from abc import ABC
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List

from fhy.ir import Type

from .core import Expression, Identifier


class UnaryOperation(StrEnum):
    """Unary (Single) Operators"""

    NEGATIVE = "-"
    BITWISE_NOT = "~"
    LOGICAL_NOT = "!"


@dataclass(frozen=True, kw_only=True)
class UnaryExpression(Expression):
    """Expressions of Unary Operators

    Args:
        _operation (UnaryOperation): Unary Operator
        _expression (Expression): An Expression the operator is performed on

    """

    operation: UnaryOperation
    expression: Expression

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_operation", "_expression"])
        return attrs

    # TODO Jason: Implement the functionality of this class


# TODO: StrEnum is Only Available for Python3.11 and Above Remake
#       Using Something Else, or Construct StrEnum if sys.version_info < (3, 11)
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
    # TODO: Add the Following Operators to Grammar (Be careful with Precedence)
    # FLOORDIV = "//"
    # MODULOS = "%"
    # POWER = "**"


@dataclass(frozen=True, kw_only=True)
class BinaryExpression(Expression):
    """Algebraic or Logical Expression Requiring Two Arguments

    Args:
        _operation (BinaryOperation): String Defining the Operation Performed
        _left_expression (Expression): Left Input Expression
        _right_expression (Expression): Right Input Expression

    """

    operation: BinaryOperation
    left: Expression
    right: Expression

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_operation", "_left_expression", "_right_expression"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class TernaryExpression(Expression):
    """A Conditional (?) Expression Node

    Args:
        _condition (Expression): Expression to Evaluate Truth
        _true_expression (Expression):
        _false_expression (Expression):

    Notes:
        {condition} ? {true Expression} : {false Expression}

    """

    condition: Expression
    true: Expression
    false: Expression

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        return attrs


@dataclass(frozen=True, kw_only=True)
class TupleAccessExpression(Expression):
    """Expression to Access Tuple Elements

    Args:
        _expression (Expression):
        _element_index (int): Tuple Index

    """

    tuple_expression: Expression
    element_index: int

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        return attrs


@dataclass(frozen=True, kw_only=True)
class FunctionExpression(Expression):
    """Function Call"""

    function: Expression
    template_types: List[Type]
    indices: List[Expression]
    args: List[Expression]

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        return attrs


@dataclass(frozen=True, kw_only=True)
class ArrayAccessExpression(Expression):
    """Tensor Indexing Node

    Args:
        _expressions (List[Expression]):
        _indices (List[Expression]):

        # NOTE: We might need Axis Arg Here

    """

    array_expression: Expression
    indices: List[Expression]

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_expressions", "_indices"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class TupleExpression(Expression):
    """Expression of a Tuple"""

    expressions: List[Expression] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_expressions"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class IdentifierExpression(Expression):
    """Unclear... Is this meant to be a declaration without value assignment?
    e.g. int i;

    """

    identifier: Identifier

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["identifier"])
        return attrs

    # TODO Jason: Implement the functionality of this class


@dataclass(frozen=True, kw_only=True)
class Literal(Expression, ABC):
    """Expression Literal Values"""


@dataclass(frozen=True, kw_only=True)
class IntLiteral(Literal):
    """Integer value Node"""

    value: int

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_value"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class FloatLiteral(Literal):
    """Floating Point value Node"""

    value: float

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_value"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class ComplexLiteral(Literal):
    """Complex Number value Node"""

    value: complex

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_value"])
        return attrs
