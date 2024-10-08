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

"""Lattice (order theory) implementation."""

from typing import Generic, TypeVar

from .poset import PartiallyOrderedSet

T = TypeVar("T")


class Lattice(Generic[T]):
    """A lattice (order theory)."""

    _poset: PartiallyOrderedSet[T]

    def __init__(self):
        self._poset = PartiallyOrderedSet[T]()

    def add_element(self, element: T) -> None:
        """Add an element to the lattice.

        Args:
            element: The element to add.

        """
        self._poset.add_element(element)

    def add_order(self, lower: T, upper: T) -> None:
        """Add an order relation between two elements.

        Args:
            lower: The lesser element.
            upper: The greater element.

        """
        self._poset.add_order(lower, upper)

    def is_lattice(self) -> bool:
        """Return True if the lattice is a valid lattice, False otherwise."""
        for x in self._poset:
            for y in self._poset:
                if not self.has_meet(x, y) or not self.has_join(x, y):
                    return False
        return True

    def has_meet(self, x: T, y: T) -> bool:
        """Check if two elements have a greatest lower bound.

        Args:
            x: The first element.
            y: The second element.

        Returns:
            bool: True if x and y have a greatest lower bound, False otherwise.

        """
        return self.get_meet(x, y) is not None

    def has_join(self, x: T, y: T) -> bool:
        """Check if two elements have a least upper bound.

        Args:
            x: The first element.
            y: The second element.

        Returns:
            bool: True if x and y have a least upper bound, False otherwise.

        """
        return self.get_join(x, y) is not None

    def get_least_upper_bound(self, x: T, y: T) -> T:
        """Get the least upper bound of two elements.

        Args:
            x: The first element.
            y: The second element.

        Returns:
            The least upper bound of x and y.

        """
        join = self.get_join(x, y)
        if join is None:
            raise RuntimeError(
                f"No least upper bound of {x} and {y} found for lattice."
            )
        return join

    def get_meet(self, x: T, y: T) -> T | None:
        """Get the greatest lower bound of two elements.

        Args:
            x: The first element.
            y: The second element.

        Returns:
            (T | None): The greatest lower bound of x and y,
                or None if it does not exist.

        """
        meet = None
        for z in self._poset:
            if self._poset.is_less_than(z, x) and self._poset.is_less_than(z, y):
                if meet is None or self._poset.is_less_than(meet, z):
                    meet = z
        return meet

    def get_join(self, x: T, y: T) -> T | None:
        """Get the least upper bound of two elements.

        Args:
            x: The first element.
            y: The second element.

        Returns:
            (T | None):
                The least upper bound of x and y, or None if it does not exist.

        """
        join = None
        for z in self._poset:
            if self._poset.is_greater_than(z, x) and self._poset.is_greater_than(z, y):
                if join is None or self._poset.is_greater_than(join, z):
                    join = z
        return join
