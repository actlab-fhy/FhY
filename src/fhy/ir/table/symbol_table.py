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

"""Symbol table and symbol table frames.

Frames (derived from SymbolTableFrame):
    ImportSymbolTableFrame: Defining any imported modules or variables
    VariableSymbolTableFrame: Defining any variables
    FunctionSymbolTableFrame: Defining any functions
"""

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NoReturn

from fhy_core import Identifier

from fhy.ir.type import Type, TypeQualifier
from fhy.utils.enumeration import StrEnum


@dataclass(frozen=True, kw_only=True)
class SymbolTableFrame(ABC):
    """Base symbol table frame.

    Args:
        name (Identifier): Variable (symbol) name and ID

    """

    name: Identifier


class ImportSymbolTableFrame(SymbolTableFrame):
    """Imported symbols are cataloged by their ID.

    Args:
        name (Identifier): Variable (symbol) name and ID

    """


@dataclass(frozen=True, kw_only=True)
class VariableSymbolTableFrame(SymbolTableFrame):
    """Variables are stored by their name, type, and type qualifier information.

    Args:
        name (Identifier): Variable (symbol) name and ID
        type (Type): variable data type
        type_qualifier (TypeQualifier): variable type qualifier information

    """

    type: Type
    type_qualifier: TypeQualifier


class FunctionKeyword(StrEnum):
    """Function keyword enumeration."""

    PROCEDURE = "proc"
    OPERATION = "op"
    NATIVE = "native"


@dataclass(frozen=True, kw_only=True)
class FunctionSymbolTableFrame(SymbolTableFrame):
    """Functions are cataloged by their argument ID and signature.

    Args:
        name (Identifier): Variable (symbol) name and ID
        keyword (FunctionKeyword): Keyword describing the kind of function.
        signature (list[Tuple[TypeQualifier, Type]]): List of arguments defined by their
            type qualifier and type, that is accepted by the function.

    """

    keyword: FunctionKeyword
    signature: list[tuple[TypeQualifier, Type]] = field(default_factory=list)


class SymbolTable:
    """Core symbol table comprised of various frames."""

    _table: dict[Identifier, dict[Identifier, SymbolTableFrame]]
    _parent_namespace: dict[Identifier, Identifier]

    def __init__(self) -> None:
        super().__init__()
        self._table = {}
        self._parent_namespace = {}

    def get_number_of_namespaces(self) -> int:
        """Retrieve the number of namespaces in the symbol table.

        Returns:
            int: Number of namespaces in the symbol table.

        """
        return len(self._table)

    def add_namespace(
        self,
        namespace_name: Identifier,
        parent_namespace_name: Identifier | None = None,
    ) -> None:
        """Add a new namespace to the symbol table.

        Args:
            namespace_name (Identifier): Name of the namespace to be added to the
                symbol table.
            parent_namespace_name (Identifier, optional): Name of the parent namespace.

        """
        self._table[namespace_name] = {}
        if parent_namespace_name:
            self._parent_namespace[namespace_name] = parent_namespace_name

    def is_namespace_defined(self, namespace_name: Identifier) -> bool:
        """Check if a namespace exists in the symbol table.

        Args:
            namespace_name (Identifier): Name of the namespace to be checked.

        Returns:
            bool: True if the namespace exists in the symbol table, False otherwise.

        """
        return namespace_name in self._table

    def get_namespace(
        self, namespace_name: Identifier
    ) -> dict[Identifier, SymbolTableFrame]:
        """Retrieve a namespace from the symbol table.

        Args:
            namespace_name (Identifier): Name of the namespace to be retrieved from
                the symbol table.

        Returns:
            Table[Identifier, SymbolTableFrame]: Namespace table.

        """
        if namespace_name not in self._table:
            raise KeyError(f"Namespace {namespace_name} not found in the symbol table.")

        return self._table[namespace_name]

    def update_namespaces(self, other_symbol_table: "SymbolTable") -> None:
        """Update the symbol table with new namespaces from another symbol table.

        Args:
            other_symbol_table (SymbolTable): Symbol table to update with.

        """
        self._table.update(other_symbol_table._table)
        self._parent_namespace.update(other_symbol_table._parent_namespace)

    def add_symbol(
        self,
        namespace_name: Identifier,
        symbol_name: Identifier,
        frame: SymbolTableFrame,
    ) -> None:
        """Add a symbol to the symbol table.

        Args:
            namespace_name (Identifier): Name of the namespace to add the symbol to.
            symbol_name (Identifier): Name of the symbol to be added.
            frame (SymbolTableFrame): Frame to be added to the symbol table.

        """
        self._table[namespace_name][symbol_name] = frame

    def is_symbol_defined(
        self, namespace_name: Identifier, symbol_name: Identifier
    ) -> bool:
        """Check if a symbol exists in the symbol table.

        Args:
            namespace_name (Identifier): Name of the namespace to check.
            symbol_name (Identifier): Name of the symbol to check.

        Returns:
            bool: True if the symbol exists in the symbol table, False otherwise.

        """

        def is_symbol_defined_in_namespace(
            namespace_name: Identifier,
        ) -> tuple[bool | None, bool]:
            if symbol_name in self._table[namespace_name]:
                return True, True
            else:
                return None, False

        return self._search_namespace_with_action(
            namespace_name, is_symbol_defined_in_namespace, lambda: False
        )

    def get_frame(
        self, namespace_name: Identifier, symbol_name: Identifier
    ) -> SymbolTableFrame:
        """Retrieve a frame from the symbol table.

        Args:
            namespace_name (Identifier): Name of the current namespace.
            symbol_name (Identifier): Name of the symbol to retrieve the frame for.

        Returns:
            SymbolTableFrame: The frame for the given symbol in the given namespace.

        """

        def get_frame_in_namespace(
            namespace_name: Identifier,
        ) -> tuple[SymbolTableFrame | None, bool]:
            if symbol_name in self._table[namespace_name]:
                return self._table[namespace_name][symbol_name], True
            else:
                return None, False

        def raise_symbol_not_found() -> NoReturn:
            # TODO: Change this to some sort of FhY error for not symbol not found
            raise RuntimeError(
                f"Symbol {symbol_name} not found in namespace {namespace_name}."
            )

        return self._search_namespace_with_action(
            namespace_name, get_frame_in_namespace, raise_symbol_not_found
        )

    def _search_namespace_with_action(
        self,
        namespace_name: Identifier,
        action: Callable[[Identifier], tuple[Any, bool]],
        action_fail_func: Callable[[], Any] = lambda: None,
    ) -> Any:
        if not self.is_namespace_defined(namespace_name):
            raise KeyError(f"Namespace {namespace_name} not found in the symbol table.")

        current_namespace_name: Identifier | None = namespace_name
        seen_namespace_names = set()
        while current_namespace_name is not None:
            if current_namespace_name in seen_namespace_names:
                raise RuntimeError(f"Namespace {current_namespace_name} is cyclic.")
            seen_namespace_names.add(current_namespace_name)

            action_ret_val, ret_code = action(current_namespace_name)
            if ret_code:
                return action_ret_val

            current_namespace_name = self._parent_namespace.get(
                current_namespace_name, None
            )

        return action_fail_func()
