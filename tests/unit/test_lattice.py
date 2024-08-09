"""Tests the lattice utility module."""

from typing import Any

import pytest
from fhy.utils.lattice import Lattice


@pytest.fixture()
def empty_lattice():
    """Return an empty lattice."""
    lattice = Lattice[Any]()
    return lattice


@pytest.fixture()
def singleton_lattice():
    """Uses the Lattice class internals to create a lattice with one element.

    lattice: ({1}, <=)

    """
    lattice = Lattice[int]()
    lattice._poset.add_element(1)
    return lattice


@pytest.fixture()
def two_element_lattice():
    """Uses the Lattice class internals to create a lattice with two elements.

    lattice: ({1, 2}, <=)

    """
    lattice = Lattice[int]()
    lattice._poset.add_element(1)
    lattice._poset.add_element(2)
    lattice._poset.add_order(1, 2)
    return lattice


def test_empty_lattice_contains_no_elements(empty_lattice: Lattice[Any]):
    """Test that an empty lattice contains no elements."""
    assert 1 not in empty_lattice


def test_singleton_lattice_contains_element(singleton_lattice: Lattice[int]):
    """Test that a singleton lattice contains the element."""
    assert 1 in singleton_lattice


def test_add_element_to_empty_lattice(empty_lattice: Lattice[Any]):
    """Test that an element can be added to an empty lattice."""
    empty_lattice.add_element(1)
    assert 1 in empty_lattice


def test_add_duplicate_element_to_lattice(singleton_lattice: Lattice[int]):
    """Test that adding a duplicate element to a lattice raises an error."""
    with pytest.raises(ValueError):
        singleton_lattice.add_element(1)


def test_singleton_lattice_meet_is_element(singleton_lattice: Lattice[int]):
    """Test that the meet of a singleton lattice is the element itself."""
    assert singleton_lattice.get_meet(1, 1) == 1


def test_singleton_lattice_has_meet(singleton_lattice: Lattice[int]):
    """Test that a singleton lattice has a meet."""
    assert singleton_lattice.has_meet(1, 1) is True


def test_singleton_lattice_join_is_element(singleton_lattice: Lattice[int]):
    """Test that the join of a singleton lattice is the element itself."""
    assert singleton_lattice.get_join(1, 1) == 1


def test_singleton_lattice_has_join(singleton_lattice: Lattice[int]):
    """Test that a singleton lattice has a join."""
    assert singleton_lattice.has_join(1, 1) is True


def test_singleton_lattice_get_least_upper_bound(singleton_lattice: Lattice[int]):
    """Test that the least upper bound of a singleton lattice is the element."""
    assert singleton_lattice.get_least_upper_bound(1, 1) == 1


def test_two_element_lattice_meet(two_element_lattice: Lattice[int]):
    """Test that the meet of a two element lattice is the lower element."""
    assert two_element_lattice.get_meet(1, 1) == 1
    assert two_element_lattice.get_meet(2, 2) == 2
    assert two_element_lattice.get_meet(1, 2) == 1


def test_two_element_lattice_has_meet(two_element_lattice: Lattice[int]):
    """Test that a two element lattice has a meet."""
    assert two_element_lattice.has_meet(1, 1) is True
    assert two_element_lattice.has_meet(2, 2) is True
    assert two_element_lattice.has_meet(1, 2) is True


def test_two_element_lattice_join(two_element_lattice: Lattice[int]):
    """Test that the join of a two element lattice is the upper element."""
    assert two_element_lattice.get_join(1, 1) == 1
    assert two_element_lattice.get_join(2, 2) == 2
    assert two_element_lattice.get_join(1, 2) == 2


def test_two_element_lattice_has_join(two_element_lattice: Lattice[int]):
    """Test that a two element lattice has a join."""
    assert two_element_lattice.has_join(1, 1) is True
    assert two_element_lattice.has_join(2, 2) is True
    assert two_element_lattice.has_join(1, 2) is True


def test_two_element_lattice_get_least_upper_bound(two_element_lattice: Lattice[int]):
    """Test that the least upper bound of a two element lattice is the element."""
    assert two_element_lattice.get_least_upper_bound(1, 1) == 1
    assert two_element_lattice.get_least_upper_bound(2, 2) == 2
    assert two_element_lattice.get_least_upper_bound(1, 2) == 2


def test_empty_lattice_meet(empty_lattice: Lattice[Any]):
    """Test that the meet of an empty lattice is None."""
    assert empty_lattice.get_meet(1, 1) is None


def test_empty_lattice_join(empty_lattice: Lattice[Any]):
    """Test that the join of an empty lattice is None."""
    assert empty_lattice.get_join(1, 1) is None


def test_add_order_to_lattice():
    """Test that an order can be added to a lattice."""
    lattice = Lattice[int]()
    lattice.add_element(1)
    lattice.add_element(2)
    lattice.add_order(1, 2)
    assert lattice.get_meet(1, 2) == 1
    assert lattice.get_join(1, 2) == 2


def test_add_invalid_order_to_lattice():
    """Test that adding an invalid order to a lattice raises an error."""
    lattice = Lattice[int]()
    lattice.add_element(1)
    lattice.add_element(2)
    lattice.add_order(1, 2)
    with pytest.raises(RuntimeError):
        lattice.add_order(2, 1)


def test_empty_lattice_is_lattice(empty_lattice: Lattice[Any]):
    """Test that an empty lattice is a lattice."""
    assert empty_lattice.is_lattice() is True


def test_singleton_lattice_is_lattice(singleton_lattice: Lattice[int]):
    """Test that a singleton lattice is a lattice."""
    assert singleton_lattice.is_lattice() is True


def test_two_element_lattice_is_lattice(two_element_lattice: Lattice[int]):
    """Test that a two element lattice is a lattice."""
    assert two_element_lattice.is_lattice() is True


@pytest.fixture()
def positive_integer_lattice():
    UPPER_BOUND = 10
    lattice = Lattice[int]()
    for i in range(1, UPPER_BOUND + 1):
        lattice.add_element(i)
    for i in range(1, UPPER_BOUND):
        lattice.add_order(i, i + 1)
    return lattice


@pytest.fixture()
def subsets_of_xyz_lattice():
    lattice = Lattice[str]()
    lattice.add_element("0")  # empty set
    lattice.add_element("x")
    lattice.add_element("y")
    lattice.add_element("z")
    lattice.add_element("xy")
    lattice.add_element("xz")
    lattice.add_element("yz")
    lattice.add_element("xyz")
    lattice.add_order("0", "x")
    lattice.add_order("0", "y")
    lattice.add_order("0", "z")
    lattice.add_order("x", "xy")
    lattice.add_order("x", "xz")
    lattice.add_order("y", "xy")
    lattice.add_order("y", "yz")
    lattice.add_order("z", "xz")
    lattice.add_order("z", "yz")
    lattice.add_order("xy", "xyz")
    lattice.add_order("xz", "xyz")
    lattice.add_order("yz", "xyz")
    return lattice


def test_positive_integer_lattice_meet(positive_integer_lattice: Lattice[int]):
    """Test that the meet of a positive integer lattice is the minimum of the
    two elements.
    """
    assert positive_integer_lattice.get_meet(3, 5) == 3
    assert positive_integer_lattice.get_meet(4, 6) == 4


def test_positive_integer_lattice_join(positive_integer_lattice: Lattice[int]):
    """Test that the join of a positive integer lattice is the maximum of the
    two elements.
    """
    assert positive_integer_lattice.get_join(3, 5) == 5
    assert positive_integer_lattice.get_join(4, 6) == 6


def test_positive_integer_lattice_is_lattice(positive_integer_lattice: Lattice[int]):
    """Test that a positive integer lattice is a lattice."""
    assert positive_integer_lattice.is_lattice() is True


def test_subsets_of_xyz_lattice_meet(subsets_of_xyz_lattice: Lattice[str]):
    """Test that the meet of subsets of XYZ lattice is the intersection of the
    two sets.
    """
    assert subsets_of_xyz_lattice.get_meet("x", "y") == "0"
    assert subsets_of_xyz_lattice.get_meet("x", "xy") == "x"
    assert subsets_of_xyz_lattice.get_meet("x", "z") == "0"
    assert subsets_of_xyz_lattice.get_meet("xz", "yz") == "z"
    assert subsets_of_xyz_lattice.get_meet("xy", "xyz") == "xy"


def test_subsets_of_xyz_lattice_join(subsets_of_xyz_lattice: Lattice[str]):
    """Test that the join of subsets of XYZ lattice is the union of the two sets."""
    assert subsets_of_xyz_lattice.get_join("x", "y") == "xy"
    assert subsets_of_xyz_lattice.get_join("x", "xy") == "xy"
    assert subsets_of_xyz_lattice.get_join("x", "z") == "xz"
    assert subsets_of_xyz_lattice.get_join("xz", "yz") == "xyz"
    assert subsets_of_xyz_lattice.get_join("xy", "xyz") == "xyz"


def test_subsets_of_xyz_lattice_is_lattice(subsets_of_xyz_lattice: Lattice[str]):
    """Test that subsets of XYZ lattice is a lattice."""
    assert subsets_of_xyz_lattice.is_lattice() is True


@pytest.fixture()
def basic_non_lattice_poset():
    lattice = Lattice[int]()
    lattice.add_element(1)
    lattice.add_element(2)
    lattice.add_element(3)
    lattice.add_element(4)
    lattice.add_order(1, 3)
    lattice.add_order(1, 4)
    lattice.add_order(2, 3)
    lattice.add_order(2, 4)
    return lattice


def test_basic_non_lattice_poset(basic_non_lattice_poset: Lattice[int]):
    """Test that a basic non-lattice poset is not a lattice."""
    assert basic_non_lattice_poset.is_lattice() is False


def test_basic_non_lattice_has_no_least_upper_bound(
    basic_non_lattice_poset: Lattice[int],
):
    """Test that a basic non-lattice poset has no least upper bound for certain
    elements.
    """
    with pytest.raises(RuntimeError):
        basic_non_lattice_poset.get_least_upper_bound(3, 4)
