"""Separate Frames of a Given Table, comprising a Symbol Table Representation.

We define separate Table Frames to catalog different types of symbols, to define and
retrieve unique symbols, by distinct attributes.

Frames (derived from SymbolTableFrame):
    ImportSymbolTableFrame: Defining any imported modules or variables
    VariableSymbolTableFrame: Defining any Variables
    FunctionSymbolTableFrame: Defining any Functions (proc, op, or native)

Tables:
    SymbolTable: The primary table used to catalog symbol variables by defined frames.

"""

from abc import ABC
from dataclasses import dataclass, field
from typing import List, Tuple

from fhy.ir.identifier import Identifier
from fhy.ir.type import Type, TypeQualifier

from .base import Table


@dataclass(frozen=True, kw_only=True)
class SymbolTableFrame(ABC):
    """Base Symbol table Frame Definition.

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


@dataclass(frozen=True, kw_only=True)
class FunctionSymbolTableFrame(SymbolTableFrame):
    """Functions are cataloged by their argument ID and signature.

    Args:
        name (Identifier): Variable (symbol) name and ID
        signature (List[Tuple[TypeQualifier, Type]]): list of arguments defined by their
            type qualifier and type, that is accepted by the function.

    """

    signature: List[Tuple[TypeQualifier, Type]] = field(default_factory=list)


class SymbolTable(Table[Identifier, Table[Identifier, SymbolTableFrame]]):
    """Core Symbol Table, comprised of various frames, mapping a symbol Identifier."""

    ...
