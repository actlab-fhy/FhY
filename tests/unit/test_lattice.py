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
    lattice.add_element("0") # empty set
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
    assert empty_lattice.is_lattice() is True, "Empty lattice is a lattice."


def test_singleton_lattice_meet_is_element(singleton_lattice: Lattice[int]):
    assert singleton_lattice.get_meet(1, 1) == 1, "Meet of a singleton lattice is the element itself."


def test_singleton_lattice_join_is_element(singleton_lattice: Lattice[int]):
    assert singleton_lattice.get_join(1, 1) == 1, "Join of a singleton lattice is the element itself."


def test_singleton_lattice_is_lattice(singleton_lattice: Lattice[int]):
    assert singleton_lattice.is_lattice() is True, "Singleton lattice is a lattice."


def test_two_element_lattice_meet(two_element_lattice: Lattice[int]):
    assert two_element_lattice.get_meet(1, 1) == 1, "Meet of lower element in two element lattice is the element itself."
    assert two_element_lattice.get_meet(2, 2) == 2, "Meet of upper element in two element lattice is the element itself."
    assert two_element_lattice.get_meet(1, 2) == 1, "Meet of two elements in two element lattice is the lower element."


def test_two_element_lattice_join(two_element_lattice: Lattice[int]):
    assert two_element_lattice.get_join(1, 1) == 1, "Join of lower element in two element lattice is the element itself."
    assert two_element_lattice.get_join(2, 2) == 2, "Join of upper element in two element lattice is the element itself."
    assert two_element_lattice.get_join(1, 2) == 2, "Join of two elements in two element lattice is the upper element."


def test_two_element_lattice_is_lattice(two_element_lattice: Lattice[int]):
    assert two_element_lattice.is_lattice() is True, "Two element lattice is a lattice."


def test_positive_integer_lattice_meet(positive_integer_lattice: Lattice[int]):
    assert positive_integer_lattice.get_meet(3, 5) == 3, "Meet of 3 and 5 in positive integer lattice is 3."
    assert positive_integer_lattice.get_meet(4, 6) == 4, "Meet of 4 and 6 in positive integer lattice is 4."


def test_positive_integer_lattice_join(positive_integer_lattice: Lattice[int]):
    assert positive_integer_lattice.get_join(3, 5) == 5, "Join of 3 and 5 in positive integer lattice is 5."
    assert positive_integer_lattice.get_join(4, 6) == 6, "Join of 4 and 6 in positive integer lattice is 6."


def test_positive_integer_lattice_is_lattice(positive_integer_lattice: Lattice[int]):
    assert positive_integer_lattice.is_lattice() is True, "Positive integer lattice is a lattice."


def test_subsets_of_xyz_lattice_meet(subsets_of_xyz_lattice: Lattice[str]):
    assert subsets_of_xyz_lattice.get_meet("x", "y") == "0", "Meet of x and y in subsets of XYZ lattice is empty set."
    assert subsets_of_xyz_lattice.get_meet("x", "xy") == "x", "Meet of x and xy in subsets of XYZ lattice is x."
    assert subsets_of_xyz_lattice.get_meet("x", "z") == "0", "Meet of x and z in subsets of XYZ lattice is empty set."
    assert subsets_of_xyz_lattice.get_meet("xz", "yz") == "z", "Meet of xz and yz in subsets of XYZ lattice is z."
    assert subsets_of_xyz_lattice.get_meet("xy", "xyz") == "xy", "Meet of xy and xyz in subsets of XYZ lattice is xy."


def test_subsets_of_xyz_lattice_join(subsets_of_xyz_lattice: Lattice[str]):
    assert subsets_of_xyz_lattice.get_join("x", "y") == "xy", "Join of x and y in subsets of XYZ lattice is xy."
    assert subsets_of_xyz_lattice.get_join("x", "xy") == "xy", "Join of x and xy in subsets of XYZ lattice is xy."
    assert subsets_of_xyz_lattice.get_join("x", "z") == "xz", "Join of x and z in subsets of XYZ lattice is xz."
    assert subsets_of_xyz_lattice.get_join("xz", "yz") == "xyz", "Join of xz and yz in subsets of XYZ lattice is xyz."
    assert subsets_of_xyz_lattice.get_join("xy", "xyz") == "xyz", "Join of xy and xyz in subsets of XYZ lattice is xyz."


def test_subsets_of_xyz_lattice_is_lattice(subsets_of_xyz_lattice: Lattice[str]):
    assert subsets_of_xyz_lattice.is_lattice() is True, "Subsets of XYZ lattice is a lattice."


def test_basic_non_lattice_poset():
    lattice = Lattice[int]()
    lattice.add_element(1)
    lattice.add_element(2)
    lattice.add_element(3)
    lattice.add_element(4)
    lattice.add_order(1, 3)
    lattice.add_order(1, 4)
    lattice.add_order(2, 3)
    lattice.add_order(2, 4)
    assert lattice.is_lattice() is False, "Basic non-lattice poset is not a lattice."
