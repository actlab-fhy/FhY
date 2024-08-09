"""Tests the stack utility module."""

import pytest
from fhy.utils.stack import Stack


@pytest.fixture
def text_stack() -> Stack:
    stack = Stack[str]()
    stack._stack.append("fhy")
    stack._stack.append("test")

    return stack


def test_empty_stack_length():
    """Test that an empty stack's length is correctly calculated."""
    stack = Stack[str]()
    assert len(stack) == 0


def test_stack_length(text_stack):
    """Test that the stack's length is correctly calculated."""
    assert len(text_stack) == 2


def test_push():
    """Test that elements are correctly pushed to the stack."""
    stack = Stack[str]()
    stack.push("fhy")
    assert len(stack) == 1
    stack.push("test")
    assert len(stack) == 2


def test_peek(text_stack):
    """Test that peek method reveals the correct element and does not mutate the
    stack.
    """
    current: str = text_stack.peek()
    assert current == "test"
    current_length: int = len(text_stack)
    assert current_length == 2, f"Expected the stack to be unchanged after peek, \
but the length changed to {current_length}."


def test_peek_error():
    """Test that stack raises an IndexError when peek is called on an empty stack."""
    stack = Stack[str]()

    with pytest.raises(IndexError):
        stack.peek()


def test_pop(text_stack):
    """Test that pop method removes the correct element from the stack."""
    first = text_stack.pop()
    assert first == "test"
    assert len(text_stack) == 1

    second = text_stack.pop()
    assert second == "fhy"
    assert len(text_stack) == 0


def test_pop_error():
    """Test that stack raises an IndexError when pop is called on an empty stack."""
    stack = Stack[str]()

    with pytest.raises(IndexError):
        stack.pop()


def test_clear(text_stack):
    """Test that clear method removes all elements from the stack."""
    text_stack.clear()
    assert len(text_stack) == 0


def test_iter(text_stack):
    """Test that we can iterate over the stack and retrieve elements."""
    for element, expected_element in zip(text_stack, ("fhy", "test")):
        assert element == expected_element

    assert len(list(text_stack)) == 2, "Expected to be able to iterate over the \
stack again."


def test_next(text_stack):
    """Test use of next on a generator of the stack to retrieve elements."""
    generator = (i for i in text_stack)

    assert next(generator) == "fhy"
    assert next(generator) == "test"
    with pytest.raises(StopIteration):
        next(generator)
