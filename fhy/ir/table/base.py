# TODO Jason: Add docstring
from typing import Dict, Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class Table(Generic[K, V]):
    _table: Dict[K, V]

    def __init__(self) -> None:
        self._table = {}

    def __getitem__(self, key: K) -> V:
        return self._table[key]

    def __setitem__(self, key: K, value: V) -> None:
        self._table[key] = value

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, key: K) -> bool:
        return key in self._table

    def keys(self):
        return self._table.keys()

    def values(self):
        return self._table.values()

    def items(self):
        return self._table.items()

    def __repr__(self) -> str:
        return f"Table({self._table})"
