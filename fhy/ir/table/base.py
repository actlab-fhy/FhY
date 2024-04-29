"""A Generic Table Mapping typed objects."""

from typing import Dict, Generic, List, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Table(Generic[K, V]):
    """A Generic table mapping typed key objects to typed value objects"""

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

    def keys(self) -> List[K]:
        """Table keys"""
        return list(self._table.keys())

    def values(self) -> List[V]:
        """Table values."""
        return list(self._table.values())

    def items(self) -> List[Tuple[K, V]]:
        """Table key, value pairs."""
        return list(self._table.items())

    def __repr__(self) -> str:
        return f"Table({self._table})"
