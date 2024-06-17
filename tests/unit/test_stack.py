"""Unit Test the stack module for basic expected behavior."""

import pytest
from fhy.utils.stack import Stack


@pytest.fixture
def text_stack() -> Stack:
    stack = Stack[str]()
    stack.push("fhy")
    stack.push("test")

    return stack


def test_stack_push(text_stack):
    """Test that we correctly add elements to the stack."""
    assert len(text_stack) == len(text_stack._stack) == 2, "Expected Length 2 in Stack."
    assert text_stack._stack[0] == "fhy", 'Expected "fhy" string at index 0'
    assert text_stack._stack[1] == "test", 'Expected "test" string at index 1'


def test_stack_peek(text_stack):
    """Test peek method reveals the correct element, and does not mutate the stack."""
    current: str = text_stack.peek()
    assert current == "test", 'Expected "test" string to be revealed'
    assert len(text_stack) == len(text_stack._stack) == 2, "Expected Length 2 in Stack."


def test_stack_peek_error():
    """Raise an Indexerror when peek is called on an Empty Stack."""
    stack = Stack[str]()

    with pytest.raises(IndexError):
        stack.peek()


def test_stack_pop(text_stack):
    """Test Element Removal."""
    first = text_stack.pop()
    assert first == "test", 'Expected "test" string to be removed'
    assert len(text_stack) == len(text_stack._stack) == 1, "Expected Length 1 in Stack."

    second = text_stack.pop()
    assert second == "fhy", 'Expected "fhy" string to be removed'
    assert len(text_stack) == len(text_stack._stack) == 0, "Expected Length 0 in Stack."


def test_stack_pop_error():
    """Attempts to pop an element from an Empty Stack should raise an Indexerror."""
    stack = Stack[str]()

    with pytest.raises(IndexError):
        stack.pop()


def test_stack_clear(text_stack):
    """Tests that clear method, removes all elements from the stack."""
    text_stack.clear()
    assert len(text_stack) == len(text_stack._stack) == 0, "Expected Length 0 in Stack."


def test_stack_iteration(text_stack):
    """Test Iteration over the stack via dunder method."""
    start: int = text_stack._iter_index
    assert start == 0, f"Starting Index should be 0: {start}"

    # NOTE: The '_iter_index' increases by 1 before returning the value
    for (index, element), value in zip(enumerate(text_stack, start=1), ("fhy", "test")):
        assert text_stack._iter_index == index, "Unexpected Index"
        assert element == value, "Unexpected Value."

    assert len(list(text_stack)) == 2, "Should be able to iterate over stack again."


def test_stack_next(text_stack):
    """Test we can use next on a generator of the stack to retrieve elements."""
    generator = (i for i in text_stack)
    assert text_stack._iter_index == 0, "Unexpected Index"

    next(generator) == "fhy"
    assert text_stack._iter_index == 1, "Unexpected Index"

    next(generator) == "test"
    assert text_stack._iter_index == 2, "Unexpected Index"

    with pytest.raises(StopIteration):
        next(generator)
