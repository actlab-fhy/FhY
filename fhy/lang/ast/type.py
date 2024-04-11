"""
    FhY/lang/ast/type.py

    Define Core AST Data Types

"""
from abc import ABC
from enum import StrEnum
from typing import List, Optional
from .base import ASTNode
from .expression import Expression


class Type(ASTNode, ABC):
    """Abstract Node Used to Define Data Types."""
    ...


# TODO: Remove StrEnum, Find another Way
class PrimitiveType(StrEnum):
    """Core Supported Primitive Types"""
    INTEGER = "int"
    FLOAT = "float"
    COMPLEX = "complex"


class DataType:
    """Data Type Defines Core Type Primitive, but of Flexible Bit Width.

    Args:
        _primitive_type: (PrimitiveType): 
        _bit_width (Optional[int]): Flexible Bit Width Specification

    """
    def __init__(self,
                 _primitive_type: PrimitiveType,
                 _bit_width: Optional[int],
                 ) -> None:
        self._primitive_type = _primitive_type
        self._bit_width = _bit_width

    # TODO Jason: Implement the functionality of this class


class NumericalType(Type):
    """Vector Array of a given DataType and Shape

    Args:
        _data_type (DataType): Type information of data contained in vector
        _shape (List[Expression]): Shape of vector

    """

    def __init__(self,
                 _data_type: DataType,
                 _shape: List[Expression]
                 ) -> None:
        super().__init__()
        self._data_type = _data_type
        self._shape = _shape

    # TODO Jason: Implement the functionality of this class
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_data_type", "_shape"])
        return attrs


class IndexType(Type):
    """An Indexer, or Slice

    Args:
        _lower_bound (Expression): start index [inclusive]
        _upper_bound (Expression): end index [inclusive]
        _stride (Optional[Expression]): increment

    Notes:
        * Grammatically similar to a python slice or range(start, stop, step)

    """
    def __init__(self,
                 _lower_bound: Expression,
                 _upper_bound: Expression,
                 _stride: Optional[Expression]
                 ) -> None:
        self._lower_bound = _lower_bound
        self._upper_bound = _upper_bound
        self._stride = _stride

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_lower_bound", "_upper_bound", "_stride"])
        return attrs

    # TODO Jason: Implement the functionality of this class


# TODO: Again, Replace Usage of StrEnum, or something.
class TypeQualifier(StrEnum):
    """Variables Have Type Qualifiers which Define Permisions."""
    # TODO Jason: Add docstring
    INPUT = "input"
    OUTPUT = "output"
    STATE = "state"
    PARAMETER = "param"
    TEMPORARY = "temp"
