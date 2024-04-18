# TODO Jason: Add docstring
from dataclasses import dataclass
from typing import Dict
from .base import ASTNode


@dataclass
class ASTNodeTypeInfo:
    fields: Dict[str, type]


_ast_node_types: Dict[type[ASTNode], ASTNodeTypeInfo] = {}


def get_ast_node_type_info(ast_node_class: type[ASTNode]) -> ASTNodeTypeInfo:
    # TODO Jason: Add docstring
    return _ast_node_types[ast_node_class]


def _get_ast_node_fields(ast_node_class: type[ASTNode]) -> Dict[str, type]:
    fields: Dict[str, type] = {}
    for cls in ast_node_class.mro():
        if ASTNode in cls.mro():
            for field_name, field_type in cls.__annotations__.items():
                fields[field_name] = field_type
    return fields


def register_ast_node(ast_node_class: type[ASTNode]) -> type[ASTNode]:
    # TODO Jason: Add docstring
    fields: Dict[str, type] = _get_ast_node_fields(ast_node_class)
    _ast_node_types[ast_node_class] = ASTNodeTypeInfo(fields)
    return ast_node_class
