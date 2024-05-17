"""Unit Test File Discovery Module Tools."""

import os
import pathlib

import pytest

from fhy.utils import discovery


@pytest.fixture
def root_dir():
    return os.path.abspath(os.path.join(__file__, os.pardir))


@pytest.fixture
def build_root(root_dir):
    yield root_dir, discovery.build_project_root(root_dir)


@pytest.fixture
def setup_traversal(build_root):
    directory, root = build_root
    parent = os.path.basename(directory)

    path = __file__
    name = os.path.basename(path)

    traversed = root.traverse_path([parent, name])

    assert isinstance(traversed, discovery.Leaf), "Expected to Receive Module Node."
    assert str(traversed.path) == path, "Expected Same Path."
    assert traversed.path.name == name, "Expected Same Name"

    yield root, traversed


def test_valid_path_standard():
    """Test valid string path is converted to absolute pathlib.Path object."""
    path = discovery.standard_path(__file__)
    assert isinstance(path, pathlib.Path), "Expected Pathlib.Path Object."


def test_invalid_type_path_standard():
    """Raise TypeError with invalid input type to standard_path function."""
    with pytest.raises(TypeError):
        discovery.standard_path(None)


def test_invalid_file_path_standard():
    """Raise TypeError with invalid input type to standard_path function."""
    with pytest.raises(FileExistsError):
        discovery.standard_path("sacagawea")


def test_root_construction(build_root):
    """Test build root."""
    directory, result = build_root

    assert isinstance(result, discovery.Root), "Expected Root Tree Node."
    assert str(result.path) == directory, "Expected Same File Path."


def test_get_named_obj(build_root):
    """Get Item."""
    _directory, result = build_root

    path = __file__
    name = os.path.basename(path)

    result = result[name]
    assert isinstance(result, discovery.Leaf), "Expected to Receive Leaf Node."


def test_descending_traversal(setup_traversal):
    """Test Downward Traversal toward leaf nodes."""
    _root, _traversed = setup_traversal


def test_ascending_traversal(setup_traversal):
    """Test Upward Traversal."""
    root, traversed = setup_traversal

    new = traversed.traverse_up(1)
    assert new.path == root.path


def test_collect_files_files_only(root_dir):
    path = os.path.join(root_dir, "data", "file_search", "A")
    extension = ".fhy"
    result = discovery.collect_files(path, extension)
    assert len(result) == 1, "Expected to find One File."
    assert all(
        i.endswith(extension) for i in result
    ), "Expected Extension argument to be Honored"


def test_collect_files_directories_too(root_dir):
    path = os.path.join(root_dir, "data", "file_search")
    extension = ".fhy"
    result = discovery.collect_files(path, extension)
    assert len(result) == 2, "Expected to find Two Files."
    assert all(
        i.endswith(extension) for i in result
    ), "Expected Extension argument to be Honored"
