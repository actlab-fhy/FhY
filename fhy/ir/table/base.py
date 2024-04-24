# TODO Jason: Add docstring
from abc import ABC, abstractmethod
from typing import Any, Dict, Union


class TableKey(ABC):
    # TODO Jason: Add docstring

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...


class TableFrame(ABC):
    # TODO Jason: Add docstring
    ...


class Table(ABC):
    # TODO Jason: Add docstring
    _table: Dict[TableKey, TableFrame]

    def __init__(self) -> None:
        self._table = {}

    def __getitem__(self, key: TableKey) -> TableFrame:
        return self._table[key]

    def __setitem__(self, key: TableKey, value: TableFrame) -> None:
        self._table[key] = value
