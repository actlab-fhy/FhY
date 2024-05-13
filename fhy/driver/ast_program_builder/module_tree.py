from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class ModuleTree(object):
    file_name: str
    parent: Optional["ModuleTree"] = field(default=None)
    children: Set["ModuleTree"] = field(default_factory=set)

    @property
    def name(self) -> str:
        current_node = self
        name_components = []
        while current_node:
            name_components.append(current_node.file_name)
            current_node = current_node.parent
        return ".".join(reversed(name_components))

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModuleTree):
            return False
        return self.name == other.name
