# TODO Jason: Add docstring
from typing import Generic, Iterator, List, TypeVar

T = TypeVar("T")


class Stack(Generic[T]):
    # TODO Jason: Add docstring
    _stack: List[T]
    _iter_index: int

    def __init__(self):
        self._stack = []
        self._iter_index = 0

    def clear(self) -> None:
        # TODO Jason: Add docstring
        self._stack.clear()
        self._iter_index = 0

    def push(self, item: T) -> None:
        # TODO Jason: Add docstring
        self._stack.append(item)

    def pop(self) -> T:
        # TODO Jason: Add docstring
        try:
            return self._stack.pop()
        except IndexError:
            raise IndexError("Cannot pop from an empty stack.")

    def peek(self) -> T:
        # TODO Jason: Add docstring
        try:
            return self._stack[-1]
        except IndexError:
            raise IndexError("Cannot peek at an empty stack.")

    def __len__(self) -> int:
        return len(self._stack)

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self._iter_index >= len(self._stack):
            raise StopIteration
        else:
            item = self._stack[self._iter_index]
            self._iter_index += 1
            return item
