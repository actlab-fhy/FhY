"""All Defined Nodes within this module are Subclasses of the core Expression ASTNode.

The Expression ASTNode is also a subclass of `fhy.ir.Expression`.

Supported Operators:
    UnaryOperation (StrEnum): Operators acting on a single expression
    BinaryOperation (StrEnum): Operators acting on two expressions

ASTNode Expressions:
    UnaryExpressions: Expressions acting on a Single expression
    BinaryOperation: Expressions acting on two expressions (left and right)
    TernaryExpression: A conditional expression, which performs true or false expression

Identities:
    IdentifierExpression: An expression Defining a Variable.

ASTNode Expression Literals:
    Literal: Abstract Expression ASTNode to define concrete values
    IntLiteral: Defines an integer value
    FloatLiteral: Defines a floating point Value
    ComplexLiteral: Defines a complex Value

"""

from abc import ABC
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List

from fhy.ir import Type

from .core import Expression, Identifier


class UnaryOperation(StrEnum):
    """Unary (Single) Operators."""

    NEGATIVE = "-"
    BITWISE_NOT = "~"
    LOGICAL_NOT = "!"


@dataclass(frozen=True, kw_only=True)
class UnaryExpression(Expression):
    """Expressions of Unary Operators.

    Args:
        operation (UnaryOperation): Supported Unary Operator
        expression (Expression): An Expression the operator is performed on

    """

    operation: UnaryOperation
    expression: Expression

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["operation", "expression"])
        return attrs


class BinaryOperation(StrEnum):
    """FhY Language Binary Operators.

    Arithmetic:
        Addition
        Subtraction
        Multiplication
        Division

    Logical:
        And
        Or

    Relational:
        Equality
        Inequality
        Less Than
        Less Than or Equal To
        Greater Than
        Greater Than or Equal To

    Bitwise:
        And
        Or
        Xor
        Left Shift
        Right Shift

    Note:
        Assignment Operators are not (yet) supported (e.g. `+=`).

    """

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
    MODULO = "%"
    POWER = "**"

    # TODO: A FloorDiv Operator looks like a Comment, and is skipped.
    # FLOORDIV = "//"


@dataclass(frozen=True, kw_only=True)
class BinaryExpression(Expression):
    """Expression Requiring Two Arguments, left and right.

    Args:
        operation (BinaryOperation): Supported Binary Operator performed
        left (Expression): Left Input Expression
        right (Expression): Right Input Expression

    """

    operation: BinaryOperation
    left: Expression
    right: Expression

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["operation", "left", "right"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class TernaryExpression(Expression):
    """A Conditional (?) Expression Node.

    Args:
        condition (Expression): Expression to Evaluate Truth
        true (Expression): Expression evaluated if true
        false (Expression): Expression evaluated if false

    Notes:
        {condition} ? {true} : {false}

    """

    condition: Expression
    true: Expression
    false: Expression

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["condition", "true", "false"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class TupleAccessExpression(Expression):
    """Expression to Access Elements within a Tuple.

    Args:
        tuple_expression (Expression): Tuple Identifier expression
        element_index (IntLiteral): Index accessed in tuple

    """

    tuple_expression: Expression
    element_index: "IntLiteral"

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["tuple_expression", "element_index"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class FunctionExpression(Expression):
    """An ASTNode expressing a Call to a Function.

    Args:
        function (Expression): Identify the function being called
        template_types (List[Type]):
        indices (List[Expression]):
        args (List[Expression]): List of user provided arguments to function call

    """

    function: Expression
    template_types: List[Type] = field(default_factory=list)
    indices: List[Expression] = field(default_factory=list)
    args: List[Expression] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["function", "template_types", "indices", "args"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class ArrayAccessExpression(Expression):
    """Array Indexing Node.

    Args:
        array_expression (Expression): Array Identifier Expression
        indices (List[Expression]): List of index access on array axes.

    """

    array_expression: Expression
    indices: List[Expression] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["array_expression", "indices"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class TupleExpression(Expression):
    """Defines a Tuple Expression of a given size."""

    expressions: List[Expression] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["expressions"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class IdentifierExpression(Expression):
    """Expresses a Variable name independent of value assignment."""

    identifier: Identifier

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["identifier"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class Literal(Expression, ABC):
    """Expression Literal defines concrete values."""


@dataclass(frozen=True, kw_only=True)
class IntLiteral(Literal):
    """Integer value ASTNode."""

    value: int

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["value"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class FloatLiteral(Literal):
    """Floating point value ASTNode."""

    value: float

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["value"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class ComplexLiteral(Literal):
    """Complex number value ASTNode."""

    value: complex

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["value"])
        return attrs
