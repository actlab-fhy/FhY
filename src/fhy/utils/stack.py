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

"""Utility to control a stack of a given type.

Classes:
    Stack: Simple typed list interface

"""

from collections import deque
from collections.abc import Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Stack(Generic[T]):
    """A simple interface to control elements within a deque (stack) of a defined type.

    Example Usage:

        .. code-block:: python

            # Instantiate the class, defining the expected type within the stack
            stack = Stack[str]()

            # Add an elements to the stack
            stack.push("fhy")
            stack.push("test")

            # View current element
            current: str = stack.peek()
            assert current == "test", "Expected `test` string in current position"

            # Remove current Element
            removed: str = stack.pop()
            assert removed == current == "test", "Unexpected Element Removed!"

            next_item = stack.peek()
            second_removed = stack.pop()
            assert next_item == second_removed == "fhy", "Unexpected Item Removed!"

            # Attempts to access or remove items from an empty stack will error
            stack.peek() # IndexError
            stack.pop()  # IndexError

    """

    _stack: deque[T]
    _iter_index: int

    def __init__(self):
        self._stack = deque[T]()
        self._iter_index = 0

    def clear(self) -> None:
        """Remove all elements within the stack."""
        self._stack.clear()
        self._iter_index = 0

    def push(self, item: T) -> None:
        """Add an item to the stack."""
        self._stack.append(item)

    def pop(self) -> T:
        """Removes a single element (right-hand) from the stack.

        Raises:
            IndexError: When function is called on an empty stack.

        """
        try:
            return self._stack.pop()

        except IndexError as e:
            raise IndexError("Cannot pop from an empty stack.") from e

    def peek(self) -> T:
        """View current (right-hand) element from the stack.

        Raises:
            IndexError: When function is called on an empty stack.

        """
        try:
            return self._stack[-1]

        except IndexError as e:
            raise IndexError("Cannot peek at an empty stack.") from e

    def __len__(self) -> int:
        return len(self._stack)

    def __iter__(self) -> Iterator[T]:
        self._iter_index = 0

        return self

    def __next__(self) -> T:
        if self._iter_index >= len(self._stack):
            raise StopIteration

        item = self._stack[self._iter_index]
        self._iter_index += 1

        return item
