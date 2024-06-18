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

from fhy import ir
from fhy.error import FhYSemanticsError
from fhy.lang.ast.node import core, expression, statement
from fhy.lang.ast.visitor import Visitor
from fhy.utils import Stack

from .identifier_collector import collect_identifiers


# TODO: when visitor class automatically visits children, remove the
#       super().visit_xxx(node) call
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

    _symbol_table: ir.SymbolTable

    _table_stack: Stack[ir.Table[ir.Identifier, ir.SymbolTableFrame]]

    def __init__(self) -> None:
        super().__init__()

        self._symbol_table = ir.SymbolTable()

        self._table_stack = Stack[ir.Table[ir.Identifier, ir.SymbolTableFrame]]()

    @property
    def symbol_table(
        self,
    ) -> ir.SymbolTable:
        return self._symbol_table

    def _push_namespace(self, namespace_name_hint: str) -> None:
        namespace_name = ir.Identifier(namespace_name_hint)
        self._symbol_table[namespace_name] = ir.Table[
            ir.Identifier, ir.SymbolTableFrame
        ]()
        self._table_stack.push(self._symbol_table[namespace_name])

    def _pop_namespace(self) -> None:
        self._table_stack.pop()

    def _assert_symbol_not_defined(self, symbol: ir.Identifier) -> None:
        if self._is_symbol_defined(symbol):
            msg = "Symbol Identifier previously declared (redefined) in current"
            raise FhYSemanticsError(f"{msg} namespace: {symbol.name_hint}")

    def _assert_symbol_defined(self, symbol: ir.Identifier) -> None:
        if not self._is_symbol_defined(symbol):
            msg = "Undeclared Symbol Identifier used in current namespace"
            raise FhYSemanticsError(f"{msg}: {symbol.name_hint}")

    def _is_symbol_defined(self, symbol: ir.Identifier) -> bool:
        for table in self._table_stack:
            if symbol in table.keys():
                return True
        return False

    def _add_symbol(self, symbol: ir.Identifier, frame: ir.SymbolTableFrame) -> None:
        if len(self._table_stack) == 0:
            raise RuntimeError(
                "Expected current table to be set before adding a symbol to it"
            )
        self._table_stack.peek()[symbol] = frame

    def __call__(self, node: core.Module, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(node, core.Module):
            raise TypeError(f'Expected a "Module" node. Received: {type(node)}')
        return super().__call__(node, *args, **kwargs)

    def visit_Module(self, node: core.Module) -> None:
        self._push_namespace(node.name.name_hint)
        super().visit_Module(node)
        self._pop_namespace()

        if len(self._table_stack) != 0:
            raise RuntimeError(
                "Expected the table stack to be empty after visiting module node."
            )

    def visit_Import(self, node: statement.Import) -> None:
        self._assert_symbol_not_defined(node.name)
        import_frame = ir.ImportSymbolTableFrame(name=node.name)
        self._add_symbol(node.name, import_frame)
        super().visit_Import(node)

    def visit_Procedure(self, node: statement.Procedure) -> None:
        self._assert_symbol_not_defined(node.name)
        proc_frame = ir.FunctionSymbolTableFrame(
            name=node.name,
            signature=[
                (arg.qualified_type.type_qualifier, arg.qualified_type.base_type)
                for arg in node.args
            ],
        )
        self._add_symbol(node.name, proc_frame)
        self._push_namespace(node.name.name_hint)
        super().visit_Procedure(node)
        self._pop_namespace()

    def visit_Operation(self, node: statement.Operation) -> None:
        self._assert_symbol_not_defined(node.name)
        op_frame = ir.FunctionSymbolTableFrame(
            name=node.name,
            signature=[
                (arg.qualified_type.type_qualifier, arg.qualified_type.base_type)
                for arg in node.args
            ]
            + [(node.return_type.type_qualifier, node.return_type.base_type)],
        )
        self._add_symbol(node.name, op_frame)
        self._push_namespace(node.name.name_hint)
        super().visit_Operation(node)
        self._pop_namespace()

    def visit_Argument(self, node: statement.Argument) -> None:
        arg_frame = ir.VariableSymbolTableFrame(
            name=node.name,
            type=node.qualified_type.base_type,
            type_qualifier=node.qualified_type.type_qualifier,
        )
        self._add_symbol(node.name, arg_frame)

        if isinstance(node.qualified_type.base_type, ir.NumericalType):
            shape_dimension_identifiers: set[ir.Identifier] = set()
            for shape in node.qualified_type.base_type.shape:
                shape_dimension_identifiers.update(collect_identifiers(shape))

            for dimension in shape_dimension_identifiers:
                if not self._is_symbol_defined(dimension):
                    var_frame = ir.VariableSymbolTableFrame(
                        name=dimension,
                        type=ir.NumericalType(ir.DataType(ir.PrimitiveDataType.UINT64)),
                        type_qualifier=ir.TypeQualifier.PARAM,
                    )
                    self._add_symbol(dimension, var_frame)

        super().visit_Argument(node)

    def visit_DeclarationStatement(self, node: statement.DeclarationStatement) -> None:
        self._assert_symbol_not_defined(node.variable_name)
        var_frame = ir.VariableSymbolTableFrame(
            name=node.variable_name,
            type=node.variable_type.base_type,
            type_qualifier=node.variable_type.type_qualifier,
        )
        self._add_symbol(node.variable_name, var_frame)
        super().visit_DeclarationStatement(node)

    def visit_IdentifierExpression(self, node: expression.IdentifierExpression) -> None:
        self._assert_symbol_defined(node.identifier)
        super().visit_IdentifierExpression(node)


def build_symbol_table(node: core.Module) -> ir.SymbolTable:
    """Build a symbol table from a module AST node.

    Argument:
        node (ast.Module): FhY module AST node

    Returns:
        (ir.SymbolTable) Symbol table cataloging all variables from the provided module,
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
