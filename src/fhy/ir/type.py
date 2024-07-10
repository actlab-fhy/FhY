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
from fhy.ir.identifier import Identifier
from fhy.utils import Lattice
from fhy.utils.enumeration import StrEnum

from .expression import Expression


class Type(ABC):
    """Abstract node defining data type."""


class DataType(ABC):
    """Abstract data type class."""


class CoreDataType(StrEnum):
    """Supported core data type primitives."""

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


def _define_uint_data_type_lattice() -> Lattice[CoreDataType]:
    lattice = Lattice[CoreDataType]()
    lattice.add_element(CoreDataType.UINT8)
    lattice.add_element(CoreDataType.UINT16)
    lattice.add_element(CoreDataType.UINT32)
    lattice.add_element(CoreDataType.UINT64)

    lattice.add_order(CoreDataType.UINT8, CoreDataType.UINT16)
    lattice.add_order(CoreDataType.UINT16, CoreDataType.UINT32)
    lattice.add_order(CoreDataType.UINT32, CoreDataType.UINT64)

    if not lattice.is_lattice():
        raise RuntimeError("Unsigned integer lattice is not a lattice.")

    return lattice


def _define_int_data_type_lattice() -> Lattice[CoreDataType]:
    lattice = Lattice[CoreDataType]()
    lattice.add_element(CoreDataType.INT8)
    lattice.add_element(CoreDataType.INT16)
    lattice.add_element(CoreDataType.INT32)
    lattice.add_element(CoreDataType.INT64)

    lattice.add_order(CoreDataType.INT8, CoreDataType.INT16)
    lattice.add_order(CoreDataType.INT16, CoreDataType.INT32)
    lattice.add_order(CoreDataType.INT32, CoreDataType.INT64)

    if not lattice.is_lattice():
        raise RuntimeError("Integer lattice is not a lattice.")

    return lattice


def _define_float_complex_data_type_lattice() -> Lattice[CoreDataType]:
    lattice = Lattice[CoreDataType]()
    lattice.add_element(CoreDataType.FLOAT16)
    lattice.add_element(CoreDataType.FLOAT32)
    lattice.add_element(CoreDataType.FLOAT64)
    lattice.add_element(CoreDataType.COMPLEX32)
    lattice.add_element(CoreDataType.COMPLEX64)
    lattice.add_element(CoreDataType.COMPLEX128)

    lattice.add_order(CoreDataType.FLOAT16, CoreDataType.FLOAT32)
    lattice.add_order(CoreDataType.FLOAT32, CoreDataType.FLOAT64)
    lattice.add_order(CoreDataType.FLOAT16, CoreDataType.COMPLEX32)
    lattice.add_order(CoreDataType.FLOAT32, CoreDataType.COMPLEX64)
    lattice.add_order(CoreDataType.FLOAT64, CoreDataType.COMPLEX128)
    lattice.add_order(CoreDataType.COMPLEX32, CoreDataType.COMPLEX64)
    lattice.add_order(CoreDataType.COMPLEX64, CoreDataType.COMPLEX128)

    if not lattice.is_lattice():
        raise RuntimeError("Floating point and complex lattice is not a lattice.")

    return lattice


_UINT_DATA_TYPE_LATTICE = _define_uint_data_type_lattice()
_INT_DATA_TYPE_LATTICE = _define_int_data_type_lattice()
_FLOAT_COMPLEX_DATA_TYPE_LATTICE = _define_float_complex_data_type_lattice()


def promote_core_data_types(
    core_data_type1: CoreDataType, core_data_type2: CoreDataType
) -> CoreDataType:
    """Promote two core data types to a common type.

    Args:
        core_data_type1 (CoreDataType): First core data type.
        core_data_type2 (CoreDataType): Second core data type.

    Returns:
        CoreDataType: Common type to which both core data types can be promoted.

    Raises:
        FhYTypeError: If the promotion is not supported.

    """
    _UINT_DATA_TYPES = {
        CoreDataType.UINT8,
        CoreDataType.UINT16,
        CoreDataType.UINT32,
        CoreDataType.UINT64,
    }
    _INT_DATA_TYPES = {
        CoreDataType.INT8,
        CoreDataType.INT16,
        CoreDataType.INT32,
        CoreDataType.INT64,
    }
    _FLOAT_COMPLEX_DATA_TYPES = {
        CoreDataType.FLOAT16,
        CoreDataType.FLOAT32,
        CoreDataType.FLOAT64,
        CoreDataType.COMPLEX32,
        CoreDataType.COMPLEX64,
        CoreDataType.COMPLEX128,
    }

    if core_data_type1 in _UINT_DATA_TYPES and core_data_type2 in _UINT_DATA_TYPES:
        return _UINT_DATA_TYPE_LATTICE.get_least_upper_bound(
            core_data_type1, core_data_type2
        )
    elif core_data_type1 in _INT_DATA_TYPES and core_data_type2 in _INT_DATA_TYPES:
        return _INT_DATA_TYPE_LATTICE.get_least_upper_bound(
            core_data_type1, core_data_type2
        )
    elif (
        core_data_type1 in _FLOAT_COMPLEX_DATA_TYPES
        and core_data_type2 in _FLOAT_COMPLEX_DATA_TYPES
    ):
        return _FLOAT_COMPLEX_DATA_TYPE_LATTICE.get_least_upper_bound(
            core_data_type1, core_data_type2
        )
    else:
        error_message: str = "Unsupported primitive data type promotion: "
        error_message += f"{core_data_type1}, {core_data_type2}"
        raise FhYTypeError(error_message)


class PrimitiveDataType(DataType):
    """Primitive data type.

    Args:
        core_data_type (CoreDataType): Core data type of the primitive data type.

    """

    _core_data_type: CoreDataType

    def __init__(self, core_data_type: CoreDataType) -> None:
        self._core_data_type = core_data_type

    @property
    def core_data_type(self) -> CoreDataType:
        """Core data type."""
        return self._core_data_type


class TemplateDataType(DataType):
    """Define template struct type.

    Args:
        data_type (Identifier): template type identifier
        widths (list[int], optional): fixed bit size width of type

    """

    _data_type: Identifier
    widths: list[int] | None

    def __init__(self, data_type: Identifier, widths: list[int] | None = None) -> None:
        self._data_type = data_type
        self.widths = widths

    @property
    def template_type(self) -> Identifier:
        """Template Identifier."""
        return self._data_type


def promote_primitive_data_types(
    primitive_data_type1: PrimitiveDataType, primitive_data_type2: PrimitiveDataType
) -> PrimitiveDataType:
    """Promote two primitive data types to a common type.

    Args:
        primitive_data_type1 (DataType): First primitive data type.
        primitive_data_type2 (DataType): Second primitive data type.

    Returns:
        DataType: Common type to which both primitive data types can be promoted.

    Raises:
        FhYTypeError: If the promotion is not supported.

    """
    return PrimitiveDataType(
        promote_core_data_types(
            primitive_data_type1.core_data_type, primitive_data_type2.core_data_type
        )
    )


class NumericalType(Type):
    """Multi-dimensional array of a given base data type and shape.

    Args:
        data_type (DataType): Base data type of multi-dimensional array.
        shape (list[Expression]): Shape of multi-dimensional array.

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
        lower_bound (Expression): Start index [inclusive].
        upper_bound (Expression): End index [inclusive].
        stride (Optional[Expression]): Step size.

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
        types (list[Type]): Types of each element within the tuple.

    """

    _types: list[Type]

    def __init__(self, types: list[Type]) -> None:
        super().__init__()
        self._types = types

    @property
    def types(self):
        return self._types


class TypeQualifier(StrEnum):
    """Supported type qualifiers to define a variable's permissions."""

    INPUT = "input"
    OUTPUT = "output"
    STATE = "state"
    PARAM = "param"
    TEMP = "temp"


def promote_type_qualifiers(
    type_qualifier1: TypeQualifier, type_qualifier2: TypeQualifier
) -> TypeQualifier:
    """Promote two type qualifiers to a common type.

    Args:
        type_qualifier1 (TypeQualifier): First type qualifier.
        type_qualifier2 (TypeQualifier): Second type qualifier.

    Returns:
        TypeQualifier: Common type to which both type qualifiers can be promoted.

    """
    if (
        type_qualifier1 == TypeQualifier.PARAM
        and type_qualifier2 == TypeQualifier.PARAM
    ):
        return TypeQualifier.PARAM
    else:
        return TypeQualifier.TEMP

    # Error message for future use if needed
    # error_message: str = "Unsupported type qualifier promotion: "
    # error_message += f"{type_qualifier1}, {type_qualifier2}"
    # raise FhYTypeError(error_message)
