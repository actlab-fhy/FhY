"""Workspace."""

from pathlib import Path


class Workspace(object):
    """Workspace."""

    _root: Path

    def __init__(self, root: Path):
        self._root = root

    @property
    def root(self) -> Path:
        return self._root
