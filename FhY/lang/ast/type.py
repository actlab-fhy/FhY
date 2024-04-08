# TODO Jason: Add docstring
from abc import ABC
from enum import StrEnum
from typing import List, Optional
from .base import ASTNode
from .expression import Expression


class Type(ASTNode, ABC):
    # TODO Jason: Add docstring
    ...


class PrimitiveType(StrEnum):
    # TODO Jason: Add docstring
    INTEGER = "int"
    FLOAT = "float"


class DataType:
    # TODO Jason: Add docstring
    _primitive_type: PrimitiveType
    _bit_width: Optional[int]

    # TODO Jason: Implement the functionality of this class


class NumericalType(Type):
    # TODO Jason: Add docstring
    _data_type: DataType
    _shape: List[Expression]

    # TODO Jason: Implement the functionality of this class


class IndexType(Type):
    # TODO Jason: Add docstring
    _lower_bound: Expression
    _upper_bound: Expression
    _stride: Optional[Expression]

    # TODO Jason: Implement the functionality of this class


class TypeQualifier(StrEnum):
    # TODO Jason: Add docstring
    INPUT = "input"
    OUTPUT = "output"
    STATE = "state"
    PARAMETER = "param"
    TEMPORARY = "temp"
