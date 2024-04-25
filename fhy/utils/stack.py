"""Utility to control a Stack of a given Type.

Classes:
    Stack: Simple typed list interface

"""

from typing import Generic, Iterator, List, TypeVar

T = TypeVar("T")


class Stack(Generic[T]):
    """A simple interface to control elements within a list (stack) of a defined type.

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

    _stack: List[T]
    _iter_index: int

    def __init__(self):
        self._stack = []
        self._iter_index = 0

    def clear(self) -> None:
        """Removes all elements within the stack"""
        self._stack.clear()
        self._iter_index = 0

    def push(self, item: T) -> None:
        """Adds an item to the stack"""
        self._stack.append(item)

    def pop(self) -> T:
        """Removes a single element (righthand) from the stack.

        Raises:
            IndexError: When function is called on an empty stack.

        """
        try:
            return self._stack.pop()
        except IndexError:
            raise IndexError("Cannot pop from an empty stack.")

    def peek(self) -> T:
        """Views the current (righthand) element from the stack.

        Raises:
            IndexError: When function is called on an empty stack.

        """
        try:
            return self._stack[-1]
        except IndexError:
            raise IndexError("Cannot peek at an empty stack.")

    def __len__(self) -> int:
        return len(self._stack)

    def __iter__(self) -> Iterator[T]:
        self._iter_index = 0
        return self

    def __next__(self) -> T:
        if self._iter_index >= len(self._stack):
            raise StopIteration
        else:
            item = self._stack[self._iter_index]
            self._iter_index += 1
            return item
