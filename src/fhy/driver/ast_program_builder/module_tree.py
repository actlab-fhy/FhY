"""Project ModuleTree path."""

from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class ModuleTree(object):
    """Module tree path.

    Args:
        file_name (str): filepath basename
        parent (Optional[ModuleTree]): Parent directory
        children (Set[ModuleTree]): Child ModuleTrees (relevant if path is a directory)

    """

    file_name: str
    parent: Optional["ModuleTree"] = field(default=None)
    children: Set["ModuleTree"] = field(default_factory=set)

    @property
    def name(self) -> str:
        """Full project filepath import name."""
        current_node: Optional[ModuleTree] = self
        name_components = []
        while current_node:
            name_components.append(current_node.file_name)
            current_node = current_node.parent

        return ".".join(reversed(name_components))

    @property
    def module_name(self) -> str:
        """Basename of module filepath."""
        return self.name.rsplit(".", maxsplit=1)[-1]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ModuleTree) and self.name == other.name
