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

"""AST pass to infer the type of a numerical expression."""

from collections.abc import Callable

from fhy.error import FhYTypeError
from fhy.ir.identifier import Identifier
from fhy.ir.type import (
    IndexType,
    Type,
    TypeQualifier,
    promote_primitive_data_types,
    promote_type_qualifiers,
)
from fhy.lang.ast.node import BinaryExpression, Expression, IdentifierExpression
from fhy.lang.ast.visitor import BasePass

_SymbolName = Identifier


class NumericalExpressionTypeInferer(BasePass):
    """Pass to infer the type of a numerical expression."""

    _is_symbol_variable: Callable[[_SymbolName], bool]
    _is_symbol_function: Callable[[_SymbolName], bool]
    _get_variable_type: Callable[[_SymbolName], tuple[Type, TypeQualifier]]
    _get_function_signature: Callable[
        [_SymbolName], tuple[list[tuple[Type, TypeQualifier]]]
    ]

    def __init__(
        self,
        is_symbol_variable_func: Callable[[_SymbolName], bool],
        is_symbol_function_func: Callable[[_SymbolName], bool],
        get_variable_type_func: Callable[[_SymbolName], tuple[Type, TypeQualifier]],
        get_function_signature_func: Callable[
            [_SymbolName], tuple[list[tuple[Type, TypeQualifier]]]
        ],
    ) -> None:
        super().__init__()
        self._is_symbol_variable = is_symbol_variable_func
        self._is_symbol_function = is_symbol_function_func
        self._get_variable_type = get_variable_type_func
        self._get_function_signature = get_function_signature_func

    def visit_BinaryExpression(
        self, node: BinaryExpression
    ) -> tuple[Type, TypeQualifier]:
        left_type, left_qualifier = self.visit(node.left)
        right_type, right_qualifier = self.visit(node.right)

        if isinstance(left_type, IndexType) or isinstance(right_type, IndexType):
            error_message: str = "Expected numerical types for AST numerical "
            error_message += f"expression type inference, but got {type(left_type)} "
            error_message += f"and {type(right_type)}."
            raise FhYTypeError(error_message)

        new_type = promote_primitive_data_types(left_type, right_type)
        new_type_qualifier = promote_type_qualifiers(left_qualifier, right_qualifier)
        return new_type, new_type_qualifier

    def visit_IdentifierExpression(
        self, node: IdentifierExpression
    ) -> tuple[Type, TypeQualifier]:
        if self._is_symbol_function(node.identifier):
            # TODO: assume that the final element in the signature is the return type
            #       The symbol table should be changed to hold this information
            #       differently.
            return self._get_function_signature(node.identifier)[-1]
        elif self._is_symbol_variable(node.identifier):
            return self._get_variable_type(node.identifier)
        else:
            error_message: str = f"Expected {node} to be a variable or function "
            error_message += "for AST numerical expression type inference."
            raise FhYTypeError(error_message)

    # TODO: JASON: Fill in the visit functions...
    # Each function should return the inferred type of the expression.


def infer_numerical_expression_type(
    node: Expression,
    is_symbol_variable_func: Callable[[_SymbolName], bool],
    is_symbol_function_func: Callable[[_SymbolName], bool],
    get_variable_type_func: Callable[[_SymbolName], tuple[Type, TypeQualifier]],
    get_function_signature_func: Callable[
        [_SymbolName], tuple[list[tuple[Type, TypeQualifier]]]
    ],
) -> tuple[Type, TypeQualifier]:
    """Infer the type of a numerical expression.

    Args:
        node (Expression): AST node representing the expression.
        is_symbol_variable_func (Callable[[Identifier], bool]):
            Function to check if a symbol is a variable.
        is_symbol_function_func (Callable[[Identifier], bool]):
            Function to check if a symbol is a function.
        get_variable_type_func
            (Callable[[Identifier], Tuple[Type, TypeQualifier]]):
            Function to get the type of a variable.
        get_function_signature_func
            (Callable[[Identifier], Tuple[List[Tuple[Type, TypeQualifier]]]):
            Function to get the signature of a function.

    Returns:
        Tuple[Type, TypeQualifier]: The inferred type of the expression.

    """
    pass_ = NumericalExpressionTypeInferer(
        is_symbol_variable_func,
        is_symbol_function_func,
        get_variable_type_func,
        get_function_signature_func,
    )
    return pass_.visit(node)
