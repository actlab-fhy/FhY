# TODO Jason: Add docstring
from typing import Any


class Identifier(object):
    """Abstracted identifier node for providing a unique ID.

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
        # TODO Jason: Add docstring
        return self._name_hint

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Identifier) and self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)

    # TODO Jason: Resolve how this identifier class can handle identifiers used in different scopes
