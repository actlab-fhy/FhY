# TODO Jason: Add docstring
from abc import ABC, abstractmethod
from ..span import Span


class ASTNode(ABC):
    _span: Span

    @abstractmethod
    def visit_attrs(self):
        raise NotImplementedError()


class Identifier(ASTNode):
    # TODO Jason: Add docstring
    _id: int
    _name_hint: str

    # TODO Jason: Implement the functionality of this class
    # TODO Jason: Resolve how this identifier class can handle identifiers used in different scopes


class Component(ASTNode, ABC):
    _name: Identifier

    # TODO Jason: Implement the functionality of this class


class Statement(ASTNode, ABC):
    # TODO Jason: Add docstring
    ...


class Expression(ASTNode, ABC):
    # TODO Jason: Add docstring
    ...
