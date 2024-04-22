# TODO Jason: Add docstring
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import List, Optional

from .expression import Expression


class Type(ABC):
    """Abstract Node Defining Data Types."""


# TODO: Remove StrEnum, Find another Way
class PrimitiveDataType(StrEnum):
    """Core Supported Primitive Types"""

    INT32 = "int32"
    FLOAT32 = "float32"


class DataType(object):
    """Data Type Defines Core Type Primitive, but of Flexible Bit Width.

    Args:
        _primitive_type: (PrimitiveType):

    """

    _primitive_data_type: PrimitiveDataType

    def __init__(
        self,
        primitive_data_type: PrimitiveDataType,
    ) -> None:
        self._primitive_data_type = primitive_data_type

    @property
    def primitive_data_type(self) -> PrimitiveDataType:
        return self._primitive_data_type

    # TODO Jason: Implement the functionality of this class


class NumericalType(Type):
    """Vector Array of a given DataType and Shape

    Args:
        _data_type (DataType): Type information of data contained in vector
        _shape (List[Expression]): Shape of vector

    """

    _data_type: DataType
    _shape: List[Expression]

    def __init__(self, data_type: DataType, shape: List[Expression]) -> None:
        super().__init__()
        self._data_type = data_type
        self._shape = shape

    @property
    def data_type(self) -> DataType:
        return self._data_type

    @property
    def shape(self) -> List[Expression]:
        return self._shape


class IndexType(Type):
    """An Indexer, or Slice

    Args:
        _lower_bound (Expression): start index [inclusive]
        _upper_bound (Expression): end index [inclusive]
        _stride (Optional[Expression]): increment

    Notes:
        * Grammatically similar to a python slice or range(start, stop, step)

    """

    def __init__(
        self,
        lower_bound: Expression,
        upper_bound: Expression,
        stride: Optional[Expression],
    ) -> None:
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._stride = stride

    @property
    def lower_bound(self) -> Expression:
        return self._lower_bound

    @property
    def upper_bound(self) -> Expression:
        return self._upper_bound

    @property
    def stride(self) -> Optional[Expression]:
        return self._stride

    # TODO Jason: Implement the functionality of this class


# TODO: Again, Replace Usage of StrEnum, or something.
class TypeQualifier(StrEnum):
    """Variables Have Type Qualifiers which Define Permisions."""

    # TODO Jason: Add docstring
    INPUT = "input"
    OUTPUT = "output"
    STATE = "state"
    PARAM = "param"
    TEMP = "temp"
