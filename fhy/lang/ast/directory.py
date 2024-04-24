"""Wrapper utilities to register ASTNode fields and their corresponding type.

Functions:
    register_ast_node: Wrapper utility to register a class node field annotations
    get_ast_node_type_info: Retrieve expected annotations of a registered node

Exceptions:
    UnregisteredASTNode


"""

from dataclasses import dataclass
from typing import Dict

from .base import ASTNode


class UnregisteredASTNode(Exception):
    """Raised when the information about a specific ASTNode type
    is not available (i.e. the ASTNode has not been registered)

    """


@dataclass
class ASTNodeTypeInfo:
    """Container class, holding class field"""

    fields: Dict[str, type]


_ast_node_types: Dict[type[ASTNode], ASTNodeTypeInfo] = {}


def get_ast_node_type_info(ast_node_class: type[ASTNode]) -> ASTNodeTypeInfo:
    """Collects Typing annotations from an ASTNode that has been previously registered.

    Raises:
        UnregisteredASTNode: When a class has not been previously registered.

    """
    try:
        return _ast_node_types[ast_node_class]

    except KeyError:
        raise UnregisteredASTNode(f"Unregistered ASTNode: {ast_node_class}")


def _get_ast_node_fields(ast_node_class: type[ASTNode]) -> Dict[str, type]:
    fields: Dict[str, type] = {}
    # NOTE: In the event a subclass overwrites an attribute, traverse the
    #       MRO backwards to prioritize the final attribute definition.
    for cls in reversed(ast_node_class.mro()):
        if ASTNode in cls.mro():
            fields.update(cls.__annotations__)
    return fields


def register_ast_node(ast_node_class: type[ASTNode]) -> type[ASTNode]:
    """Registers an ASTNode Attribute Field Names and Type Information"""
    fields: Dict[str, type] = _get_ast_node_fields(ast_node_class)
    _ast_node_types[ast_node_class] = ASTNodeTypeInfo(fields)
    return ast_node_class
