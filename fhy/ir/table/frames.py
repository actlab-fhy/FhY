from abc import ABC
from dataclasses import dataclass, field
from typing import List, Tuple, Set
from fhy.ir.identifier import Identifier
from fhy.ir.type import Type, TypeQualifier


@dataclass(frozen=True, kw_only=True)
class SymbolTableFrame(ABC):
    name: Identifier


@dataclass(frozen=True, kw_only=True)
class ImportSymbolTableFrame(SymbolTableFrame):
    name: Identifier


@dataclass(frozen=True, kw_only=True)
class VariableSymbolTableFrame(SymbolTableFrame):
    type: Type
    type_qualifier: TypeQualifier


@dataclass(frozen=True, kw_only=True)
class FunctionSymbolTableFrame(SymbolTableFrame):
    signature: List[Tuple[TypeQualifier, Type]] = field(default_factory=list)


# @dataclass(frozen=True, kw_only=True)
# class NamespaceTableFrame(object):
#     name: Identifier
#     defining_construct_name: Identifier
#     parent_namespace: Identifier
#     child_namespaces: Set[Identifier]
