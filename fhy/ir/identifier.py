"""Variable Identifier Class Object to assign Unique ID"""

from typing import Any


class Identifier(object):
    """Identifier node to assign a unique ID.

    Args:
        name_hint (str): Variable Name

    """

    _next_id: int = 0
    _id: int
    _name_hint: str

    def __init__(self, name_hint: str) -> None:
        super().__init__()
        self._id = Identifier._next_id
        Identifier._next_id += 1
        self._name_hint = name_hint

    @property
    def name_hint(self) -> str:
        """Variable Name"""
        return self._name_hint

    @property
    def id(self) -> int:
        """UniqueID"""
        return self._id

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Identifier) and self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)

    def __repr__(self) -> str:
        return f"Identifier({self._name_hint}::{self._id})"
