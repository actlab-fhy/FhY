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

"""Methods to construct a symbol table from an AST node.

Functions:
    build_symbol_table: Primary entry point to construct a symbol table from an ASTnode.

Classes:
    SymbolTableBuilder: The workhorse behind building a symbol table from AST

"""

from typing import Any

from fhy.error import FhYSemanticsError
from fhy.ir.builtins import BUILTIN_LANG_IDENTIFIERS, BUILTINS_NAMESPACE_NAME
from fhy.ir.identifier import Identifier
from fhy.ir.table import (
    FunctionKeyword,
    FunctionSymbolTableFrame,
    ImportSymbolTableFrame,
    SymbolTable,
    SymbolTableFrame,
    VariableSymbolTableFrame,
)
from fhy.ir.type import CoreDataType, NumericalType, PrimitiveDataType, TypeQualifier
from fhy.lang.ast.node import core, expression, statement
from fhy.lang.ast.visitor import Visitor
from fhy.utils import Stack

from .identifier_collector import collect_identifiers


class SymbolTableBuilder(Visitor):
    """Builds a symbol table for the given AST module node.

    The class will throw an exception if a variable is used before being declared or if
    a variable is declared more than once within the same namespace.

    Note:
        This builder pass for the symbol table only supports namespaces created by
        a new module or new operation/procedure. Nested scopes created by ForAll
        loop bodies and If/Else bodies are not supported and will be treated as
        the same namespace as the parent operation/procedure.

    Raises:
        FhYSemanticsError: A variable is used before being declared (undefined), or
            the variable is defined again (redefined), within the current namespace.
        RuntimeError: Unexpected behavior, indicating improper use.
        TypeError: Received wrong argument (node) type.

    """

    _symbol_table: SymbolTable

    _namespace_stack: Stack[Identifier]

    def __init__(self) -> None:
        super().__init__()

        self._symbol_table = SymbolTable()
        self._symbol_table.add_namespace(BUILTINS_NAMESPACE_NAME, None)
        for identifier in BUILTIN_LANG_IDENTIFIERS.values():
            self._symbol_table.add_symbol(
                BUILTINS_NAMESPACE_NAME,
                identifier,
                ImportSymbolTableFrame(name=identifier),
            )

        self._namespace_stack = Stack[Identifier]()
        self._namespace_stack.push(BUILTINS_NAMESPACE_NAME)

    @property
    def symbol_table(
        self,
    ) -> SymbolTable:
        return self._symbol_table

    def _push_namespace(self, namespace_name: Identifier) -> None:
        if len(self._namespace_stack) == 0:
            parent_namespace_name = None
        else:
            parent_namespace_name = self._namespace_stack.peek()

        self._symbol_table.add_namespace(namespace_name, parent_namespace_name)
        self._namespace_stack.push(namespace_name)

    def _pop_namespace(self) -> None:
        self._namespace_stack.pop()

    def _assert_symbol_not_defined(self, symbol: Identifier) -> None:
        if self._is_symbol_defined(symbol):
            msg = "Symbol Identifier previously declared (redefined) in current"
            raise FhYSemanticsError(f"{msg} namespace: {symbol.name_hint}")

    def _assert_symbol_defined(self, symbol: Identifier) -> None:
        if not self._is_symbol_defined(symbol):
            msg = "Undeclared Symbol Identifier used in current namespace"
            raise FhYSemanticsError(f"{msg}: {symbol.name_hint}")

    def _is_symbol_defined(self, symbol: Identifier) -> bool:
        return self._symbol_table.is_symbol_defined(
            self._namespace_stack.peek(), symbol
        )

    def _add_symbol(self, symbol: Identifier, frame: SymbolTableFrame) -> None:
        if len(self._namespace_stack) == 0:
            raise RuntimeError(
                "Expected current namespace to be set before adding a symbol to it."
            )
        self._symbol_table.add_symbol(self._namespace_stack.peek(), symbol, frame)

    def __call__(self, node: core.Module, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(node, core.Module):
            raise TypeError(f'Expected a "Module" node. Received: {type(node)}')
        return super().__call__(node, *args, **kwargs)

    def visit_Module(self, node: core.Module) -> None:
        self._push_namespace(node.name)
        super().visit_Module(node)
        self._pop_namespace()

        if len(self._namespace_stack) != 1:
            error_message: str = "Expected the namespace stack to only contain "
            error_message += f"{BUILTINS_NAMESPACE_NAME} after visiting "
            error_message += "module node."
            raise RuntimeError(error_message)

    def visit_Import(self, node: statement.Import) -> None:
        self._assert_symbol_not_defined(node.name)
        import_frame = ImportSymbolTableFrame(name=node.name)
        self._add_symbol(node.name, import_frame)
        super().visit_Import(node)

    def visit_Procedure(self, node: statement.Procedure) -> None:
        self._assert_symbol_not_defined(node.name)
        proc_frame = FunctionSymbolTableFrame(
            name=node.name,
            keyword=FunctionKeyword.PROCEDURE,
            signature=[
                (arg.qualified_type.type_qualifier, arg.qualified_type.base_type)
                for arg in node.args
            ],
        )
        self._add_symbol(node.name, proc_frame)
        self._push_namespace(node.name)
        super().visit_Procedure(node)
        self._pop_namespace()

    def visit_Operation(self, node: statement.Operation) -> None:
        self._assert_symbol_not_defined(node.name)
        op_frame = FunctionSymbolTableFrame(
            name=node.name,
            keyword=FunctionKeyword.OPERATION,
            signature=[
                (arg.qualified_type.type_qualifier, arg.qualified_type.base_type)
                for arg in node.args
            ]
            + [(node.return_type.type_qualifier, node.return_type.base_type)],
        )
        self._add_symbol(node.name, op_frame)
        self._push_namespace(node.name)
        super().visit_Operation(node)
        self._pop_namespace()

    def visit_Argument(self, node: statement.Argument) -> None:
        arg_frame = VariableSymbolTableFrame(
            name=node.name,
            type=node.qualified_type.base_type,
            type_qualifier=node.qualified_type.type_qualifier,
        )
        self._add_symbol(node.name, arg_frame)

        if isinstance(node.qualified_type.base_type, NumericalType):
            shape_dimension_identifiers: set[Identifier] = set()
            for shape in node.qualified_type.base_type.shape:
                shape_dimension_identifiers.update(collect_identifiers(shape))

            for dimension in shape_dimension_identifiers:
                if not self._is_symbol_defined(dimension):
                    var_frame = VariableSymbolTableFrame(
                        name=dimension,
                        type=NumericalType(PrimitiveDataType(CoreDataType.UINT64)),
                        type_qualifier=TypeQualifier.PARAM,
                    )
                    self._add_symbol(dimension, var_frame)

        super().visit_Argument(node)

    def visit_DeclarationStatement(self, node: statement.DeclarationStatement) -> None:
        self._assert_symbol_not_defined(node.variable_name)
        var_frame = VariableSymbolTableFrame(
            name=node.variable_name,
            type=node.variable_type.base_type,
            type_qualifier=node.variable_type.type_qualifier,
        )
        self._add_symbol(node.variable_name, var_frame)
        super().visit_DeclarationStatement(node)

    def visit_IdentifierExpression(self, node: expression.IdentifierExpression) -> None:
        self._assert_symbol_defined(node.identifier)
        super().visit_IdentifierExpression(node)


def build_symbol_table(node: core.Module) -> SymbolTable:
    """Build a symbol table from a module AST node.

    Argument:
        node (ast.Module): FhY module AST node

    Returns:
        (SymbolTable) Symbol table cataloging all variables from the provided module,
            by appropriate frame.

    Raises:
        FhYSemanticsError: A variable is used before being declared (undefined), or
            the variable is defined again (redefined), within the current namespace.
        RuntimeError: Unexpected behavior, indicating improper use.
        TypeError: Received wrong argument (node) type.

    """
    builder = SymbolTableBuilder()
    builder(node)

    return builder.symbol_table
