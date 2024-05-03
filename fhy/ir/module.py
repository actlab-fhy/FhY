"""IR Module Node."""

from dataclasses import dataclass
from typing import Set

from .identifier import Identifier


@dataclass(frozen=True, kw_only=True)
class Module(object):
    """IR Module Node."""

    name: Identifier
    parent: "Module"
    children: Set["Module"]
