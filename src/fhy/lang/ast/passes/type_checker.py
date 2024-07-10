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

"""AST pass to check the type of a module."""

from collections.abc import Callable

from fhy.ir.identifier import Identifier
from fhy.ir.type import (
    Type,
    TypeQualifier,
)
from fhy.lang.ast.visitor import Visitor
from fhy.utils import Stack

_SymbolName = Identifier
_ScopeName = Identifier


class TypeChecker(Visitor):
    """Pass to check the type of a module."""

    _is_symbol_variable: Callable[[_SymbolName, _ScopeName], bool]
    _is_symbol_proc: Callable[[_SymbolName, _ScopeName], bool]
    _is_symbol_op: Callable[[_SymbolName, _ScopeName], bool]
    _get_variable_type: Callable[[_SymbolName, _ScopeName], tuple[Type, TypeQualifier]]
    _get_function_signature: Callable[
        [_SymbolName, _ScopeName], tuple[list[tuple[Type, TypeQualifier]]]
    ]

    _scope_stack: Stack[_ScopeName]

    def __init__(
        self,
        is_symbol_variable_func: Callable[[_SymbolName, _ScopeName], bool],
        is_symbol_proc_func: Callable[[_SymbolName, _ScopeName], bool],
        is_symbol_op_func: Callable[[_SymbolName, _ScopeName], bool],
        get_variable_type_func: Callable[
            [_SymbolName, _ScopeName], tuple[Type, TypeQualifier]
        ],
        get_function_signature_func: Callable[
            [_SymbolName, _ScopeName], tuple[list[tuple[Type, TypeQualifier]]]
        ],
    ) -> None:
        super().__init__()
        self._is_symbol_variable = is_symbol_variable_func
        self._is_symbol_proc = is_symbol_proc_func
        self._is_symbol_op = is_symbol_op_func
        self._get_variable_type = get_variable_type_func
        self._get_function_signature = get_function_signature_func

        self._scope_stack = Stack[_ScopeName]()

    # TODO: Jason: Fill this in!
