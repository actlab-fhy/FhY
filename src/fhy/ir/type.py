"""Data type node definitions."""

from abc import ABC
from enum import StrEnum
from typing import List, Optional

from .expression import Expression


class Type(ABC):
    """Abstract node defining data type."""


class PrimitiveDataType(StrEnum):
    """Supported primitive data types."""

    INT = "int"
    FLOAT = "float"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class DataType(object):
    """Data type defines core type primitive, but of flexible Bit Width.

    Note:
        Currently, only supports primitive data types and does not support
        template types.

    Args:
        primitive_data_type (PrimitiveType):

    """

    _primitive_data_type: PrimitiveDataType

    def __init__(
        self,
        primitive_data_type: PrimitiveDataType,
    ) -> None:
        self._primitive_data_type = primitive_data_type

    @property
    def primitive_data_type(self) -> PrimitiveDataType:
        """Primitive data type."""
        return self._primitive_data_type

    def __repr__(self) -> str:
        return f"DataType({self._primitive_data_type})"


class NumericalType(Type):
    """Vector array of a given DataType and shape.

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
    """An indexer, or slice.

    Args:
        lower_bound (Expression): Start index [inclusive]
        upper_bound (Expression): End index [inclusive]
        stride (Optional[Expression]): Increment

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
    """Tuple data type.

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
