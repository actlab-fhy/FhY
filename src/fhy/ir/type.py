# Copyright (c) 2024 FhY Developers
# Christopher Priebe <cpriebe@ucsd.edu>
# Jason C Del Rio <j3delrio@ucsd.edu>
# Hadi S Esmaeilzadeh <hadi@ucsd.edu>
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be
# used to endorse or promote products derived from this software without specific prior
# written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""FhY type system definitions."""

from abc import ABC

from fhy.error import FhYTypeError
from fhy.utils import Lattice
from fhy.utils.enumeration import StrEnum

from .expression import Expression


class Type(ABC):
    """Abstract node defining data type."""


class PrimitiveDataType(StrEnum):
    """Supported primitive data types."""

    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    COMPLEX32 = "complex32"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"


def _define_uint_data_type_lattice() -> Lattice[PrimitiveDataType]:
    lattice = Lattice[PrimitiveDataType]()
    lattice.add_element(PrimitiveDataType.UINT8)
    lattice.add_element(PrimitiveDataType.UINT16)
    lattice.add_element(PrimitiveDataType.UINT32)
    lattice.add_element(PrimitiveDataType.UINT64)

    lattice.add_order(PrimitiveDataType.UINT8, PrimitiveDataType.UINT16)
    lattice.add_order(PrimitiveDataType.UINT16, PrimitiveDataType.UINT32)
    lattice.add_order(PrimitiveDataType.UINT32, PrimitiveDataType.UINT64)

    if not lattice.is_lattice():
        raise RuntimeError("Unsigned integer lattice is not a lattice.")

    return lattice


def _define_int_data_type_lattice() -> Lattice[PrimitiveDataType]:
    lattice = Lattice[PrimitiveDataType]()
    lattice.add_element(PrimitiveDataType.INT8)
    lattice.add_element(PrimitiveDataType.INT16)
    lattice.add_element(PrimitiveDataType.INT32)
    lattice.add_element(PrimitiveDataType.INT64)

    lattice.add_order(PrimitiveDataType.INT8, PrimitiveDataType.INT16)
    lattice.add_order(PrimitiveDataType.INT16, PrimitiveDataType.INT32)
    lattice.add_order(PrimitiveDataType.INT32, PrimitiveDataType.INT64)

    if not lattice.is_lattice():
        raise RuntimeError("Integer lattice is not a lattice.")

    return lattice


def _define_float_complex_data_type_lattice() -> Lattice[PrimitiveDataType]:
    lattice = Lattice[PrimitiveDataType]()
    lattice.add_element(PrimitiveDataType.FLOAT16)
    lattice.add_element(PrimitiveDataType.FLOAT32)
    lattice.add_element(PrimitiveDataType.FLOAT64)
    lattice.add_element(PrimitiveDataType.COMPLEX32)
    lattice.add_element(PrimitiveDataType.COMPLEX64)
    lattice.add_element(PrimitiveDataType.COMPLEX128)

    lattice.add_order(PrimitiveDataType.FLOAT16, PrimitiveDataType.FLOAT32)
    lattice.add_order(PrimitiveDataType.FLOAT32, PrimitiveDataType.FLOAT64)
    lattice.add_order(PrimitiveDataType.FLOAT16, PrimitiveDataType.COMPLEX32)
    lattice.add_order(PrimitiveDataType.FLOAT32, PrimitiveDataType.COMPLEX64)
    lattice.add_order(PrimitiveDataType.FLOAT64, PrimitiveDataType.COMPLEX128)
    lattice.add_order(PrimitiveDataType.COMPLEX32, PrimitiveDataType.COMPLEX64)
    lattice.add_order(PrimitiveDataType.COMPLEX64, PrimitiveDataType.COMPLEX128)

    if not lattice.is_lattice():
        raise RuntimeError("Floating point and complex lattice is not a lattice.")

    return lattice


_UINT_DATA_TYPE_LATTICE = _define_uint_data_type_lattice()
_INT_DATA_TYPE_LATTICE = _define_int_data_type_lattice()
_FLOAT_COMPLEX_DATA_TYPE_LATTICE = _define_float_complex_data_type_lattice()


def promote_primitive_data_types(
    primitive_data_type1: PrimitiveDataType, primitive_data_type2: PrimitiveDataType
) -> PrimitiveDataType:
    _UINT_DATA_TYPES = {
        PrimitiveDataType.UINT8,
        PrimitiveDataType.UINT16,
        PrimitiveDataType.UINT32,
        PrimitiveDataType.UINT64,
    }
    _INT_DATA_TYPES = {
        PrimitiveDataType.INT8,
        PrimitiveDataType.INT16,
        PrimitiveDataType.INT32,
        PrimitiveDataType.INT64,
    }
    _FLOAT_COMPLEX_DATA_TYPES = {
        PrimitiveDataType.FLOAT16,
        PrimitiveDataType.FLOAT32,
        PrimitiveDataType.FLOAT64,
        PrimitiveDataType.COMPLEX32,
        PrimitiveDataType.COMPLEX64,
        PrimitiveDataType.COMPLEX128,
    }

    if (
        primitive_data_type1 in _UINT_DATA_TYPES
        and primitive_data_type2 in _UINT_DATA_TYPES
    ):
        return _UINT_DATA_TYPE_LATTICE.get_least_upper_bound(
            primitive_data_type1, primitive_data_type2
        )
    elif (
        primitive_data_type1 in _INT_DATA_TYPES
        and primitive_data_type2 in _INT_DATA_TYPES
    ):
        return _INT_DATA_TYPE_LATTICE.get_least_upper_bound(
            primitive_data_type1, primitive_data_type2
        )
    elif (
        primitive_data_type1 in _FLOAT_COMPLEX_DATA_TYPES
        and primitive_data_type2 in _FLOAT_COMPLEX_DATA_TYPES
    ):
        return _FLOAT_COMPLEX_DATA_TYPE_LATTICE.get_least_upper_bound(
            primitive_data_type1, primitive_data_type2
        )
    else:
        raise FhYTypeError(
            f"Unsupported primitive data type promotion: {primitive_data_type1}, {primitive_data_type2}"
        )


class DataType:
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


def promote_data_types(data_type1: DataType, data_type2: DataType) -> DataType:
    return DataType(
        promote_primitive_data_types(
            data_type1.primitive_data_type, data_type2.primitive_data_type
        )
    )


class NumericalType(Type):
    """Vector array of a given DataType and shape.

    Args:
        data_type (DataType): Type information of data contained in vector
        shape (list[Expression]): Shape of vector

    """

    _data_type: DataType
    _shape: list[Expression]

    def __init__(
        self, data_type: DataType, shape: list[Expression] | None = None
    ) -> None:
        super().__init__()
        self._data_type = data_type
        self._shape = shape or []

    @property
    def data_type(self) -> DataType:
        return self._data_type

    @property
    def shape(self) -> list[Expression]:
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
    _stride: Expression | None

    def __init__(
        self,
        lower_bound: Expression,
        upper_bound: Expression,
        stride: Expression | None,
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
    def stride(self) -> Expression | None:
        return self._stride

    def __repr__(self) -> str:
        return f"IndexType({self._lower_bound}, {self._upper_bound}, {self._stride})"


class TupleType(Type):
    """Tuple data type.

    Args:
        types (list[Type]): types of each element within the tuple

    """

    _types: list[Type]

    def __init__(self, types: list[Type]) -> None:
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


def promote_type_qualifiers(
    type_qualifier1: TypeQualifier, type_qualifier2: TypeQualifier
) -> TypeQualifier:
    if (
        type_qualifier1 == TypeQualifier.PARAM
        and type_qualifier2 == TypeQualifier.PARAM
    ):
        return TypeQualifier.PARAM
    else:
        return TypeQualifier.TEMP

    # Error message for future use if needed
    # raise FhYTypeError(f"Unsupported type qualifier promotion: {type_qualifier1}, {type_qualifier2}")
