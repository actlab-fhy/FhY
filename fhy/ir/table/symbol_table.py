from abc import ABC
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict
from fhy.ir.identifier import Identifier
from fhy.ir.type import Type, TypeQualifier
from .base import Table


@dataclass(frozen=True, kw_only=True)
class SymbolTableFrame(ABC):
    name: Identifier


class ImportSymbolTableFrame(SymbolTableFrame):
    ...


@dataclass(frozen=True, kw_only=True)
class VariableSymbolTableFrame(SymbolTableFrame):
    type: Type
    type_qualifier: TypeQualifier


@dataclass(frozen=True, kw_only=True)
class FunctionSymbolTableFrame(SymbolTableFrame):
    signature: List[Tuple[TypeQualifier, Type]] = field(default_factory=list)


class SymbolTable(Table[Identifier, Table[Identifier, SymbolTableFrame]]):
    ...
