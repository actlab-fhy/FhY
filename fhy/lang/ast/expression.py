# TODO Jason: Add docstring
from abc import ABC
from enum import StrEnum
from typing import List

from fhy.ir import Type

from .core import Expression, Identifier


class UnaryOperation(StrEnum):
    """Unary (Single) Operators"""

    NEGATIVE = "-"
    BITWISE_NOT = "~"
    LOGICAL_NOT = "!"


class UnaryExpression(Expression):
    """Expressions of Unary Operators

    Args:
        _operation (UnaryOperation): Unary Operator
        _expression (Expression): An Expression the operator is performed on

    """

    def __init__(
        self,
        _operation: UnaryOperation,
        _expression: Expression,
    ) -> None:
        super().__init__()
        self._operation = _operation
        self._expression = _expression

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


class BinaryExpression(Expression):
    """Algebraic or Logical Expression Requiring Two Arguments

    Args:
        _operation (BinaryOperation): String Defining the Operation Performed
        _left_expression (Expression): Left Input Expression
        _right_expression (Expression): Right Input Expression

    """

    def __init__(
        self,
        _operation: BinaryOperation,
        _left_expression: Expression,
        _right_expression: Expression,
    ) -> None:
        super().__init__()
        self._operation = _operation
        self._left_expression = _left_expression
        self._right_expression = _right_expression

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_operation", "_left_expression", "_right_expression"])
        return attrs


class TernaryExpression(Expression):
    """A Conditional (?) Expression Node

    Args:
        _condition (Expression): Expression to Evaluate Truth
        _true_expression (Expression):
        _false_expression (Expression):

    Notes:
        {condition} ? {true Expression} : {false Expression}

    """

    def __init__(
        self,
        _condition: Expression,
        _true_expression: Expression,
        _false_expression: Expression,
    ) -> None:
        super().__init__()
        self._condition = _condition
        self._true_expression = _true_expression
        self._false_expression = _false_expression

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        return attrs


class TupleAccessExpression(Expression):
    """Expression to Access Tuple Elements

    Args:
        _expression (Expression):
        _element_index (int): Tuple Index

    """

    def __init__(self, _expression: Expression, _element_index: int) -> None:
        super().__init__()
        self._expression = _expression
        self._element_index = _element_index

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        return attrs


class FunctionExpression(Expression):
    """Option"""

    def __init__(
        self,
        _template_types: List[Type],
        _indices: List[Expression],
        _args: List[Expression],
        _return_type: Type,
    ) -> None:
        super().__init__()
        self._template_types = _template_types
        self._indices = _indices
        self._args = _args
        self._return_type = _return_type

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        return attrs


class TensorAccessExpression(Expression):
    """Tensor Indexing Node

    Args:
        _expressions (List[Expression]):
        _indices (List[Expression]):

        # NOTE: We might need Axis Arg Here

    """

    def __init__(
        self, _expressions: List[Expression], _indices: List[Expression]
    ) -> None:
        super().__init__()
        self._expressions = _expressions
        self._indices = _indices

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_expressions", "_indices"])
        return attrs


class TupleExpression(Expression):
    """Expression of a Tuple"""

    def __init__(self, _expressions: List[Expression]) -> None:
        super().__init__()
        self._expressions = _expressions

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_expressions"])
        return attrs


class IdentifierExpression(Expression):
    """Unclear... Is this meant to be a declaration without value assignment?
    e.g. int i;

    """

    def __init__(self, _identifier: Identifier) -> None:
        super().__init__()
        self._identifier = _identifier

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_identifier"])
        return attrs

    # TODO Jason: Implement the functionality of this class


class Literal(Expression, ABC):
    """Expression Literal Values"""


class IntLiteral(Literal):
    """Integer value Node"""

    def __init__(self, value: int) -> None:
        self._value = value

    @property
    def value(self) -> int:
        """Value of Integer"""
        return self._value

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_value"])
        return attrs


class FloatLiteral(Literal):
    """Floating Point value Node"""

    def __init__(self, value: float) -> None:
        self._value = value

    @property
    def value(self) -> float:
        """Value of Floating Point number"""
        return self._value

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_value"])
        return attrs


class ComplexLiteral(Literal):
    """Complex Number value Node"""

    _value: complex

    def __init__(self, value: complex) -> None:
        self._value = value

    @property
    def value(self) -> complex:
        """Value of complex number"""
        return self._value

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_value"])
        return attrs
