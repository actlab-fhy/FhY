from typing import Generic, TypeVar

from .poset import POSet

T = TypeVar("T")


class Lattice(Generic[T]):
    """A lattice (order theory)."""

    _poset: POSet[T]

    def __init__(self):
        self._poset = POSet[T]()

    def add_element(self, element: T):
        """Add an element to the lattice."""
        self._poset.add_element(element)

    def add_order(self, lower: T, upper: T):
        """Add an order relation between two elements."""
        self._poset.add_order(lower, upper)

    def is_lattice(self) -> bool:
        """Check if the poset is a lattice."""
        for x in self._poset:
            for y in self._poset:
                if not self.has_meet(x, y) or not self.has_join(x, y):
                    return False
        return True

    def has_meet(self, x: T, y: T) -> bool:
        """Check if two elements have a greatest lower bound."""
        return self.get_meet(x, y) is not None

    def has_join(self, x: T, y: T) -> bool:
        """Check if two elements have a least upper bound."""
        return self.get_join(x, y) is not None

    def get_least_upper_bound(self, x: T, y: T) -> T:
        """Get the least upper bound of two elements."""
        join = self.get_join(x, y)
        if join is None:
            raise RuntimeError(
                f"No least upper bound of {x} and {y} found for lattice."
            )
        return join

    def get_meet(self, x: T, y: T) -> T | None:
        """Get the greatest lower bound of two elements."""
        meet = None
        for z in self._poset:
            if self._poset.is_less_than(z, x) and self._poset.is_less_than(z, y):
                if meet is None or self._poset.is_less_than(meet, z):
                    meet = z
        return meet

    def get_join(self, x: T, y: T) -> T | None:
        """Get the least upper bound of two elements."""
        join = None
        for z in self._poset:
            if self._poset.is_greater_than(z, x) and self._poset.is_greater_than(z, y):
                if join is None or self._poset.is_greater_than(join, z):
                    join = z
        return join
