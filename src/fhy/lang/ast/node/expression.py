# Copyright (c) 2024 FhY Developers
# Christopher Priebe <cpriebe@ucsd.edu>
# Jason C Del Rio <j3delrio@ucsd.edu>
# Hadi S Esmaeilzadeh <hadi@ucsd.edu>
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be
# used to endorse or promote products derived from this software without specific prior
# written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""Expression nodes for the expressions in the FhY language.

Supported Operators:
    UnaryOperation (StrEnum): Operators acting on a single expression.
    BinaryOperation (StrEnum): Operators acting on two expressions.

AST Expressions:
    UnaryExpressions: Expression acting on a single expression.
    BinaryOperation: Expression acting on two expressions (left and right).
    TernaryExpression: Conditional expression.

Primitive Expressions:
    TupleExpression: An expression defining a tuple.
    TupleAccessExpression: An expression accessing a tuple element.
    FunctionExpression: An expression defining a function call.
    ArrayAccessExpression: An expression accessing an array element.
    IdentifierExpression: An expression defining a variable.
    Literal: Abstract expression node to define concrete values.
    IntLiteral: Defines an integer value.
    FloatLiteral: Defines a floating point value.
    ComplexLiteral: Defines a complex value.

"""

from abc import ABC
from dataclasses import dataclass, field

from fhy_core import Identifier

from fhy.ir.type import DataType as IRDataType
from fhy.utils.enumeration import StrEnum

from .core import Expression


class UnaryOperation(StrEnum):
    """FhY language unary operators.

    Arithmetic:
        Negation

    Logical:
        Logical Not

    Bitwise:
        Bitwise Not

    """

    NEGATION = "-"
    BITWISE_NOT = "~"
    LOGICAL_NOT = "!"


@dataclass(frozen=True, kw_only=True)
class UnaryExpression(Expression):
    """Expression node representing expressions requiring one argument.

    Args:
        operation (UnaryOperation): Supported unary operator.
        expression (Expression): Input expression.

    Attributes:
        operation (UnaryOperation): Supported unary operator.
        expression (Expression): Input expression.

    """

    operation: UnaryOperation
    expression: Expression

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["operation", "expression"])

        return attrs


class BinaryOperation(StrEnum):
    """FhY language binary operators.

    Arithmetic:
        Addition
        Subtraction
        Multiplication
        Division
        Floor Division
        Modulo
        Power

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

    """

    MULTIPLICATION = "*"
    DIVISION = "/"
    ADDITION = "+"
    SUBTRACTION = "-"
    LEFT_SHIFT = "<<"
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
    FLOORDIV = "//"


@dataclass(frozen=True, kw_only=True)
class BinaryExpression(Expression):
    """Expression node representing expressions requiring two arguments.

    Args:
        operation (BinaryOperation): Binary operator.
        left (Expression): Left-input expression.
        right (Expression): Right-input expression.

    Attributes:
        operation (BinaryOperation): Binary operator.
        left (Expression): Left-input expression.
        right (Expression): Right-input expression.

    """

    operation: BinaryOperation
    left: Expression
    right: Expression

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["operation", "left", "right"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class TernaryExpression(Expression):
    """Conditional (?) expression node.

    Args:
        condition (Expression): Input expression evaluated as a boolean.
        true (Expression): Input expression if condition is true.
        false (Expression): Input expression if condition is false.

    Attributes:
        condition (Expression): Input expression evaluated as a boolean.
        true (Expression): Input expression if condition is true.
        false (Expression): Input expression if condition is false.

    """

    condition: Expression
    true: Expression
    false: Expression

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["condition", "true", "false"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class TupleAccessExpression(Expression):
    """Expression node representing access to a tuple element.

    Args:
        tuple_expression (Expression): Expression defining the tuple.
        element_index (IntLiteral): Index of the element to access.

    Attributes:
        tuple_expression (Expression): Expression defining the tuple.
        element_index (IntLiteral): Index of the element to access.

    """

    tuple_expression: Expression
    element_index: "IntLiteral"

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["tuple_expression", "element_index"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class FunctionExpression(Expression):
    """Function call expression node.

    Args:
        function (Expression): Expression defining the function to call.
        template_types (List[IRDataType], optional): Data types for template arguments.
        indices (List[Expression], optional): Reduced indices for a reduction
            operation.
        args (List[Expression], optional): Provided arguments to function call.

    Attributes:
        function (Expression): Expression defining the function to call.
        template_types (List[IRDataType]): Types for template arguments.
        indices (List[Expression]): Reduced indices for a reduction operation.
        args (List[Expression]): Provided arguments to function call.

    """

    function: Expression
    template_types: list[IRDataType] = field(default_factory=list)
    indices: list[Expression] = field(default_factory=list)
    args: list[Expression] = field(default_factory=list)

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["function", "template_types", "indices", "args"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class ArrayAccessExpression(Expression):
    """Array access expression node.

    Args:
        array_expression (Expression): Expression defining the array.
        indices (List[Expression], optional): Indices to access the array.

    Attributes:
        array_expression (Expression): Expression defining the array.
        indices (List[Expression]): Indices to access the array.

    """

    array_expression: Expression
    indices: list[Expression] = field(default_factory=list)

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["array_expression", "indices"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class TupleExpression(Expression):
    """Tuple expression node.

    Args:
        expressions (List[Expression], optional): Expressions defining the
            tuple.

    Attributes:
        expressions (List[Expression]): Expressions defining the tuple.

    """

    expressions: list[Expression] = field(default_factory=list)

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["expressions"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class IdentifierExpression(Expression):
    """Wrapper node for a variable is used in an expression.

    Args:
        identifier (Identifier): Identifier of the variable.

    Attributes:
        identifier (Identifier): Identifier of the variable.

    """

    identifier: Identifier

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["identifier"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class Literal(Expression, ABC):
    """Abstract expression node to define concrete values."""


@dataclass(frozen=True, kw_only=True)
class IntLiteral(Literal):
    """Expression node for integer literals.

    Args:
        value (int): Integer value.

    Attributes:
        value (int): Integer value.

    """

    value: int

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["value"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class FloatLiteral(Literal):
    """Expression node for floating point literals.

    Args:
        value (float): Floating point value.

    Attributes:
        value (float): Floating point value.

    """

    value: float

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["value"])

        return attrs


@dataclass(frozen=True, kw_only=True)
class ComplexLiteral(Literal):
    """Expression node for complex literals.

    Args:
        value (complex): Complex value.

    Attributes:
        value (complex): Complex value.

    """

    value: complex

    def get_visit_attrs(self) -> list[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["value"])

        return attrs
