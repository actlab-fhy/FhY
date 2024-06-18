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

import networkx as nx

T = TypeVar("T")


class PartiallyOrderedSet(Generic[T]):
    """A partially ordered set (poset)."""

    _graph: nx.DiGraph

    def __init__(self):
        self._graph = nx.DiGraph()

    def __iter__(self):
        return iter(nx.topological_sort(self._graph))

    def __len__(self) -> int:
        return len(self._graph.nodes)

    def add_element(self, element: T) -> None:
        """Add an element to the poset.

        Args:
            element: The element to add.

        """
        self._graph.add_node(element)

    def add_order(self, lower: T, upper: T) -> None:
        """Add an order relation between two elements.

        Args:
            lower: The lesser element.
            upper: The greater element.

        """
        if not self._graph.has_node(lower) or not self._graph.has_node(upper):
            raise RuntimeError(
                "Expected nodes to be added to the poset before adding an order."
            )
        if nx.has_path(self._graph, upper, lower):
            error_message: str = f"Expected {lower} to be less than {upper} "
            error_message += "in the poset, but there is already a path "
            error_message += f"{upper} to {lower}."
            raise RuntimeError(error_message)
        self._graph.add_edge(lower, upper)

    def is_less_than(self, lower: T, upper: T) -> bool:
        """Check if one element is less than another.

        Args:
            lower: The postulated lesser element.
            upper: The postulated greater element.

        Returns (bool): True if lower is less than upper, False otherwise.
        """
        return nx.has_path(self._graph, lower, upper)

    def is_greater_than(self, lower: T, upper: T) -> bool:
        """Check if one element is greater than another.

        Args:
            lower: The postulated greater element.
            upper: The postulated lesser element.

        Returns (bool): True if lower is greater than upper, False otherwise.

        """
        return nx.has_path(self._graph, upper, lower)
