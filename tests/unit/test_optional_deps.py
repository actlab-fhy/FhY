"""Tests the optional dependency import utility functions."""

from unittest.mock import MagicMock, patch

import pytest
from fhy.utils.optional_deps import ErrorLevel, get_version, import_optional_dependency


def mock_module_with_version(name: str = "mock_module", version: str | None = None):
    """Create a mock module with a version attribute."""
    mock_module = MagicMock()
    mock_module.__name__ = name
    if version:
        mock_module.__version__ = version
    return mock_module


@pytest.mark.parametrize("version", ["1.0.0", "2.3.0", "3.0.2"])
def test_get_version_with_version(version: str):
    """Test that the get_version function returns the correct version."""
    module = mock_module_with_version(version=version)
    assert get_version(module) == version


def test_get_version_without_version():
    """Test that the get_version function raises ImportError when version is missing."""
    module = mock_module_with_version()
    with pytest.raises(ImportError):
        get_version(module)


@patch("importlib.import_module")
def test_import_optional_dependency_valid(mock_import_module):
    """Test that import_optional_dependency returns the module when it is available."""
    name = "mock_module"
    mock_module = mock_module_with_version(name=name, version="2.3.0")
    mock_import_module.return_value = mock_module

    module = import_optional_dependency(name)
    assert module == mock_module
    mock_import_module.assert_called_once_with(name)


@patch("importlib.import_module", side_effect=ImportError)
def test_import_optional_dependency_missing(mock_import_module):
    """Test that import_optional_dependency returns None when the module is missing."""
    with pytest.raises(ImportError):
        import_optional_dependency("mock_module", error_level=ErrorLevel.RAISE)


# TODO: Uncomment when WARN is implemented
# @patch("importlib.import_module", side_effect=ImportError)
# def test_import_optional_dependency_missing_warn(mock_import_module):
#     """Test that import_optional_dependency returns None when the module
# is missing."""
#     module = import_optional_dependency("mock_module", error_level=ErrorLevel.WARN)
#     assert module is None


@patch("importlib.import_module", side_effect=ImportError)
def test_import_optional_dependency_missing_ignore(mock_import_module):
    """Test that import_optional_dependency returns None when the module is missing."""
    module = import_optional_dependency("mock_module", error_level=ErrorLevel.IGNORE)
    assert module is None


@patch("importlib.import_module")
def test_import_optional_dependency_version_too_old(mock_import_module):
    """Test that import_optional_dependency raises an ImportError when the
    version is too old.
    """
    mock_module = mock_module_with_version(name="mock_module", version="2.0.0")
    mock_import_module.return_value = mock_module

    with pytest.raises(ImportError):
        import_optional_dependency(
            "mock_module", min_version="2.3.0", error_level=ErrorLevel.RAISE
        )


# TODO: Uncomment when WARN is implemented
# @patch("importlib.import_module")
# def test_import_optional_dependency_version_warn(mock_import_module):
#     """Test that import_optional_dependency returns None when the version is
# too old."""
#     mock_module = mock_module_with_version(name="mock_module", version="2.0.0")
#     mock_import_module.return_value = mock_module

#     module = import_optional_dependency("mock_module", min_version="2.3.0",
# error_level=ErrorLevel.WARN)
#     assert module is None


@patch("importlib.import_module")
def test_import_optional_dependency_valid_version(mock_import_module):
    """Test that import_optional_dependency returns the module when the version
    is valid.
    """
    mock_module = mock_module_with_version(name="mock_module", version="2.3.1")
    mock_import_module.return_value = mock_module

    module = import_optional_dependency("mock_module", min_version="2.3.0")
    assert module == mock_module
