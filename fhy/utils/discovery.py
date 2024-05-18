"""Utilities to discover and construct a Directory Tree."""

import os
import pathlib
import warnings
from collections.abc import Mapping
from typing import Dict, Generator, List, Optional, Self, Tuple, Union


# TODO: Include an `ignore` argument, which uses regex to filter filepaths
def collect_files(
    directory: str,
    extension: Optional[Union[Tuple[str, ...], str]] = None,
) -> List[str]:
    """Collect all child filepaths from a directory, recursively.

    Args:
        directory (str): valid directory or filepath
        extension (Optional[str]): If provided, find files with extension(s)

    Returns:
        (List[str]) list of all filepaths disseminating from directory

    """
    files: List[str] = []
    for f in os.scandir(directory):
        if f.is_dir():
            # NOTE: We may have Rules about Subfolders. e.g.
            # root = os.path.join(f.path, "__root__.fhy")
            # if os.path.exists(root): ...
            files.extend(collect_files(f.path))
            continue

        elif f.is_file() and f.path != ".DS_Store":
            if extension is None or f.path.endswith(extension):
                files.append(f.path)

        else:
            warnings.warn(f"Unknown File Path Type: {f.path}")

    return files


def collect_paths(root: Union[str, pathlib.Path]) -> List[pathlib.Path]:
    """Return a list of pathlib objects for children of root directory."""
    return [pathlib.Path(i.path) for i in os.scandir(root)]


def standard_path(value: Union[str, pathlib.Path]) -> pathlib.Path:
    """Standardize and resolve file path.

    Raises:
        TypeError: Provided type of value is not of type AnyPath (str | pathlib.Path)
        FileExistsError: Provided Path does not exist on file system.

    """
    if isinstance(value, str):
        path = pathlib.Path(value)

    elif isinstance(value, pathlib.Path):
        path = value

    else:
        msg = "Expected `value` argument of type `str` | `pathlib.Path`."
        raise TypeError(f"{msg} Received: {value}")

    if not path.exists():
        raise FileExistsError(f"Filepath Does Not Exist: {value}")

    return path.resolve()


# TODO: We want to be able to define rules on expected files, or files to ignore, etc...
# TODO: Do we have an expectation of subdirectories (e.g. python must have an
#       __init__.py file to be considered included as part of the package.)
class Root(Mapping):
    """Project Root Node, mapping filepaths in a directory tree structure.

    Args:
        path (str | pathlib.Path): Provide a valid filepath directory

    Attributes:
        tree (Dict[str, AParentLeaf | Leaf]): mapping of names to child paths.

    Raises:
        TypeError: received invalid path argument type
        FileExistsError: path (or child filepath) does not exist
        KeyError: name does not exist in mapping

    Usage:

        .. code-block::

            parent_directory = os.path.join(__file__, os.pardir)
            root = Root(parent)

            # populate the tree with child leaf nodes
            root.get_children()

    """

    tree: Dict[str, Union["AParentLeaf", "Leaf"]]
    _path: pathlib.Path

    def __init__(self, path: Union[str, pathlib.Path]) -> None:
        super().__init__()
        self.path = standard_path(path)
        self.tree = {}

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @path.setter
    def path(self, value: Union[str, pathlib.Path]):
        temp = standard_path(value)
        if not temp.is_dir():
            warnings.warn(f"Root Node Path should be a Directory: {temp}")
        self._path = temp

    def get_children(self) -> None:
        """Populate Directory Tree recursively with child nodes."""
        for p in collect_paths(self.path):
            if p.is_dir():
                result = AParentLeaf(p, self)
                result.get_children()
                self.tree[p.name] = result

            elif p.is_file():
                self.tree[p.name] = Leaf(p, self)

            else:
                warnings.warn(f"Unknown File Path Type: {p}")

    def traverse_path(self, path: List[str]) -> "Root":
        """Traverse down toward child leaf nodes."""
        if len(path) > 1:
            return self[path[0]].traverse_path(path[1:])

        elif len(path) == 1:
            return self[path[0]]

        else:
            return self

    def traverse_up(self, levels: int):
        """Traverse from Leaf Nodes toward root."""
        if isinstance(self, (AParentLeaf, Leaf)) and levels > 0:
            return self.parent.traverse_up(levels - 1)
        return self

    def __getitem__(self, key: str) -> "Root":
        if self.path.name == key:
            return self

        return self.tree[key]

    # def __setitem__(self, key: str, value: "Root") -> None:
    #     self.tree[key] = value

    def __iter__(self) -> Generator[str, None, None]:
        yield from self.tree

    def __len__(self) -> int:
        return len(self.tree)


class AParentLeaf(Root):
    """Child Directory and bad Pun."""

    def __init__(self, path: Union[str, pathlib.Path], parent: Root) -> None:
        super().__init__(path)
        self.parent = parent


class Leaf(Root):
    """Leaf file, or module Node."""

    def __init__(self, path: Union[str, pathlib.Path], parent: Root) -> None:
        super().__init__(path)
        self.parent = parent

    @Root.path.setter
    def path(self, value: Union[str, pathlib.Path]):
        temp = standard_path(value)
        if not temp.is_file():
            warnings.warn(f"Leaf Node Path should be a File: {temp}")
        self._path = temp

    def traverse_path(self, path: List[str]) -> Self:
        return self

    def get_children(self) -> None: ...


def build_project_root(directory: str) -> Root:
    """Construct a File Directory Tree from a directory path."""
    root = Root(directory)
    root.get_children()

    return root
