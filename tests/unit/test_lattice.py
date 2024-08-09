"""Tests the lattice utility module."""

from typing import Any

import pytest
from fhy.utils.lattice import Lattice


@pytest.fixture()
def empty_lattice():
    lattice = Lattice[Any]()
    return lattice


@pytest.fixture()
def singleton_lattice():
    lattice = Lattice[int]()
    lattice.add_element(1)
    return lattice


@pytest.fixture()
def two_element_lattice():
    lattice = Lattice[int]()
    lattice.add_element(1)
    lattice.add_element(2)
    lattice.add_order(1, 2)
    return lattice


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


def test_empty_lattice_is_lattice(empty_lattice: Lattice[Any]):
    """Test that an empty lattice is a lattice."""
    assert empty_lattice.is_lattice() is True


def test_singleton_lattice_meet_is_element(singleton_lattice: Lattice[int]):
    """Test that the meet of a singleton lattice is the element itself."""
    assert singleton_lattice.get_meet(1, 1) == 1


def test_singleton_lattice_join_is_element(singleton_lattice: Lattice[int]):
    """Test that the join of a singleton lattice is the element itself."""
    assert singleton_lattice.get_join(1, 1) == 1


def test_singleton_lattice_is_lattice(singleton_lattice: Lattice[int]):
    """Test that a singleton lattice is a lattice."""
    assert singleton_lattice.is_lattice() is True


def test_two_element_lattice_meet(two_element_lattice: Lattice[int]):
    """Test that the meet of a two element lattice is the lower element."""
    assert two_element_lattice.get_meet(1, 1) == 1
    assert two_element_lattice.get_meet(2, 2) == 2
    assert two_element_lattice.get_meet(1, 2) == 1


def test_two_element_lattice_join(two_element_lattice: Lattice[int]):
    """Test that the join of a two element lattice is the upper element."""
    assert two_element_lattice.get_join(1, 1) == 1
    assert two_element_lattice.get_join(2, 2) == 2
    assert two_element_lattice.get_join(1, 2) == 2


def test_two_element_lattice_is_lattice(two_element_lattice: Lattice[int]):
    """Test that a two element lattice is a lattice."""
    assert two_element_lattice.is_lattice() is True


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


def test_basic_non_lattice_poset():
    """Test that a basic non-lattice poset is not a lattice."""
    lattice = Lattice[int]()
    lattice.add_element(1)
    lattice.add_element(2)
    lattice.add_element(3)
    lattice.add_element(4)
    lattice.add_order(1, 3)
    lattice.add_order(1, 4)
    lattice.add_order(2, 3)
    lattice.add_order(2, 4)
    assert lattice.is_lattice() is False
