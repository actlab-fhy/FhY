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

"""Partially ordered set (poset) implementation."""

from typing import Generic, TypeVar

import networkx as nx  # type: ignore

T = TypeVar("T")


class PartiallyOrderedSet(Generic[T]):
    """A partially ordered set (poset)."""

    _graph: nx.DiGraph

    def __init__(self):
        self._graph = nx.DiGraph()

    def __contains__(self, element: T) -> bool:
        return self._graph.has_node(element)

    def __iter__(self):
        return iter(nx.topological_sort(self._graph))

    def __len__(self) -> int:
        return len(self._graph.nodes)

    def add_element(self, element: T) -> None:
        """Add an element to the poset.

        Args:
            element: The element to add.

        Raises:
            ValueError: If the element is already a member of the poset.

        """
        self._check_element_not_in_poset(element)
        self._graph.add_node(element)

    def add_order(self, lower: T, upper: T) -> None:
        """Add an order relation between two elements.

        Args:
            lower: The lesser element.
            upper: The greater element.

        Raises:
            ValueError: If either lower or upper is not a member of the poset.
            RuntimeError: If an order relation already exists between lower and upper.

        """
        self._check_element_in_poset(lower)
        self._check_element_in_poset(upper)
        if nx.has_path(self._graph, upper, lower):
            raise RuntimeError(
                f"Expected no order between {lower} and {upper}, but found one."
            )
        self._graph.add_edge(lower, upper)

    def is_less_than(self, lower: T, upper: T) -> bool:
        """Check if one element is less than another.

        Args:
            lower: The postulated lesser element.
            upper: The postulated greater element.

        Returns:
            bool: True if lower is less than upper, False otherwise.

        Raises:
            ValueError: If either lower or upper is not a member of the poset.

        """
        self._check_element_in_poset(lower)
        self._check_element_in_poset(upper)
        return nx.has_path(self._graph, lower, upper)

    def is_greater_than(self, lower: T, upper: T) -> bool:
        """Check if one element is greater than another.

        Args:
            lower: The postulated greater element.
            upper: The postulated lesser element.

        Returns:
            bool: True if lower is greater than upper, False otherwise.

        Raises:
            ValueError: If either lower or upper is not a member of the poset.

        """
        self._check_element_in_poset(lower)
        self._check_element_in_poset(upper)
        return nx.has_path(self._graph, upper, lower)

    def _check_element_not_in_poset(self, element: T) -> None:
        if element in self:
            raise ValueError(
                f"Expected {element} to not be a member of the poset, but it is."
            )

    def _check_element_in_poset(self, element: T) -> None:
        if element not in self:
            raise ValueError(
                f"Expected {element} to be a member of the poset, but it is not."
            )
