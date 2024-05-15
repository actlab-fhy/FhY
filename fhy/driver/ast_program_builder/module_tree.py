"""Project Module Tree Path."""

from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class ModuleTree(object):
    """Module Tree Path.

    Args:
        file_name (str): filepath basename
        parent (Optional[ModuleTree]): Parent Directory Module Tree
        children (Set[ModuleTree]): Child Module Trees (relevant if file is a directory)

    """

    file_name: str
    parent: Optional["ModuleTree"] = field(default=None)
    children: Set["ModuleTree"] = field(default_factory=set)

    @property
    def name(self) -> str:
        """Full Project Filepath Import Name."""
        current_node: Optional[ModuleTree] = self
        name_components = []
        while current_node:
            name_components.append(current_node.file_name)
            current_node = current_node.parent

        return ".".join(reversed(name_components))

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ModuleTree) and self.name == other.name
