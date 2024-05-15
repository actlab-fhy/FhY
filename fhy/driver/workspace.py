"""Workspace Root Path."""

from pathlib import Path


class Workspace(object):
    """Workspace Describing Project Root Main File Path.

    Args:
        root (Path): Path to main FhY Filepath in src directory

    Usage:
        If we have the following project diagram:
        .. code-block:: text

            Root/
            └── Src/
                ├── main.fhy
                ├── other_module.fhy
                └── subpackage/
                    └── submodule.fhy

        Then we point to the main module (entry point) within the src directory:
        .. code-block:: python

            path = Path("Root/Src/main.fhy")
            workspace = Workspace(path)

    """

    _root: Path

    def __init__(self, root: Path):
        self._root = root

    @property
    def main(self) -> Path:
        """Path Indicating Primary Entry Point Module of Project."""
        return self._root

    @property
    def source(self) -> Path:
        """Parent Source Directory Containing Project."""
        return self._root.parent

    @property
    def root(self) -> Path:
        """Root Directory Encapsulating Source."""
        return self.source.parent
