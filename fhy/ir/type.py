"""Data Type Node Definitions."""

from abc import ABC
from enum import StrEnum
from typing import List, Optional

from .expression import Expression


class Type(ABC):
    """Abstract Node Defining Data Type."""


class PrimitiveDataType(StrEnum):
    """Supported Primitive Data Types."""

    INT = "int"

    INT32 = "int32"
    FLOAT32 = "float32"


class DataType(object):
    """Data Type Defines Core Type Primitive, but of Flexible Bit Width.

    Args:
        primitive_data_type: (PrimitiveType):

    """

    _primitive_data_type: PrimitiveDataType

    def __init__(
        self,
        primitive_data_type: PrimitiveDataType,
    ) -> None:
        self._primitive_data_type = primitive_data_type

    @property
    def primitive_data_type(self) -> PrimitiveDataType:
        """Primitive Data Type"""
        return self._primitive_data_type

    def __repr__(self) -> str:
        return f"DataType({self._primitive_data_type})"


class NumericalType(Type):
    """Vector Array of a given DataType and Shape.

    Args:
        data_type (DataType): Type information of data contained in vector
        shape (List[Expression]): Shape of vector

    """

    _data_type: DataType
    _shape: List[Expression]

    def __init__(
        self, data_type: DataType, shape: Optional[List[Expression]] = None
    ) -> None:
        super().__init__()
        self._data_type = data_type
        self._shape = shape or []

    @property
    def data_type(self) -> DataType:
        return self._data_type

    @property
    def shape(self) -> List[Expression]:
        return self._shape

    def __repr__(self) -> str:
        shape = ",".join(repr(s) for s in self._shape)
        return f"NumericalType({self._data_type}, [{shape}])"


class IndexType(Type):
    """An Indexer, or Slice

    Args:
        lower_bound (Expression): start index [inclusive]
        upper_bound (Expression): end index [inclusive]
        stride (Optional[Expression]): increment

    Notes:
        * Grammatically similar to a python slice or range(start, stop, step)

    """

    _lower_bound: Expression
    _upper_bound: Expression
    _stride: Optional[Expression]

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

    def __repr__(self) -> str:
        return f"IndexType({self._lower_bound}, {self._upper_bound}, {self._stride})"


class TupleType(Type):
    """Tuple Data Type

    Args:
        types (List[Type]): types of each element within the tuple

    """

    _types: List[Type]

    def __init__(self, types: List[Type]) -> None:
        super().__init__()
        self._types = types

    @property
    def types(self):
        return self._types


class TypeQualifier(StrEnum):
    """Supported type qualifiers define a variable's permissions."""

    INPUT = "input"
    OUTPUT = "output"
    STATE = "state"
    PARAM = "param"
    TEMP = "temp"
