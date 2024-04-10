# TODO Jason: Add docstring
from abc import ABC, abstractmethod
from typing import Any, List
from .identifier import Identifier
from ..span import Span


class ASTNode(ABC):
    """Core Abstract AST Node"""
    _span: Span

    @classmethod
    def keyname(cls) -> str:
        """Class Node Name"""
        if hasattr(cls, "__qualname__"):
            return cls.__qualname__
        return cls.__name__

    @abstractmethod
    def visit_attrs(self) -> List[str]:
        ...


class Component(ASTNode, ABC):
    """Component Node"""
    _name: Identifier

    def __init__(self, name: Identifier) -> None:
        super().__init__()
        self._name = name

    @property
    def name(self) -> Identifier:
        return self._name

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        return ["_name"]


class Module(ASTNode):
    _components: List[Component]

    def __init__(self, *args: Component) -> None:
        super().__init__()
        self._components = list(args)

    @property
    def components(self) -> List[Component]:
        return self._components.copy()

    def add_component(self, component: Component) -> None:
        self._components.append(component)

    def visit_attrs(self) -> List[str]:
        return ["_components"]


class Statement(ASTNode, ABC):
    """Abstract Statement Node"""

    def visit_attrs(self) -> List[str]:
        return super().visit_attrs()


class Expression(ASTNode, ABC):
    """Abstract Expression Definition"""

    def visit_attrs(self) -> List[str]:
        return super().visit_attrs()
