""" """
from typing import Any, Dict, Type

from ..ast.base import ASTNode
from ..ast.directory import get_ast_node_type_info, ASTNodeTypeInfo


class FieldAttributeError(Exception):
    """Raised when Attempt to Assign an Unsupported Attribute."""


class ASTBuilderFrame:
    """Core Builder Class Constructor.

    Args:
        node_class (Type[ASTNode]): Registered ASTNode cls
        **kwargs: Supported Keyword Arguments of the ASTNode

    Raises:
        UnregisteredASTNode: When Provided 
        FieldAttributeError:

    """
    _protected = ("_node_class", "_type_info")

    def __init__(self, node_class: Type[ASTNode], **kwargs) -> None:
        self._node_class: Type[ASTNode] = node_class
        self._type_info: ASTNodeTypeInfo = get_ast_node_type_info(self._node_class)

        self._attributes = set(list(self._type_info.fields.keys()))
        self.update(**kwargs)
        self.setup()

    @property
    def __annotations__(self) -> Dict[str, type]:
        self._type_info.fields

    def setup(self):
        """Assign undefined attributes of the Class to None."""
        for key in self._attributes:
            if not hasattr(self, key):
                setattr(self, key, None)

    def build(self) -> ASTNode:
        """Builds the Desired Class using the collected attributes."""
        data = {k: getattr(self, k, None) for k in self._attributes}
        return self._node_class(**data)

    def update(self, **kwargs):
        """Updates and/or Overwrites the Class Attributes"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        if not hasattr(self, "_attributes"):
            ...
        elif name not in self._attributes and name not in self._protected:
            raise FieldAttributeError(f"Unsupported Attribute Assignment: {name}")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        self._attributes.remove(name)
