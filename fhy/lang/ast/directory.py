# TODO Jason: Add docstring
from dataclasses import dataclass
from typing import Dict
from .base import ASTNode


class UnregisteredASTNode(Exception):
    """Raised when the information about a specific ASTNode type
    is not available (i.e. the ASTNode has not been registered)

    """
    ...


@dataclass
class ASTNodeTypeInfo:
    fields: Dict[str, type]


_ast_node_types: Dict[type[ASTNode], ASTNodeTypeInfo] = {}


def get_ast_node_type_info(ast_node_class: type[ASTNode]) -> ASTNodeTypeInfo:
    """Collects Typing Information from an ASTNode that has been previously registered.

    Raises:
        UnregisteredASTNode: When a class has not been previously registered.

    """
    try:
        return _ast_node_types[ast_node_class]

    except KeyError:
        raise UnregisteredASTNode(f"Unregistered ASTNode: {ast_node_class}")


def _get_ast_node_fields(ast_node_class: type[ASTNode]) -> Dict[str, type]:
    fields: Dict[str, type] = {}
    # NOTE: In the event a Subclass Overwrites an attribute, traverse the
    #       MRO Backwards, to prioritize the final Atribute Definition
    for cls in reversed(ast_node_class.mro()):
        if ASTNode in cls.mro():
            fields.update(cls.__annotations__)
    return fields


def register_ast_node(ast_node_class: type[ASTNode]) -> type[ASTNode]:
    """Registers an ASTNode Attribute Field Names and Type Information"""
    fields: Dict[str, type] = _get_ast_node_fields(ast_node_class)
    _ast_node_types[ast_node_class] = ASTNodeTypeInfo(fields)
    return ast_node_class
