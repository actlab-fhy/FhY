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

"""Data type node definitions."""

from abc import ABC

from fhy.ir.identifier import Identifier as _Identifier
from fhy.utils.enumeration import StrEnum

from .expression import Expression


class Type(ABC):
    """Abstract node defining data type."""


class DataType(ABC):
    """Abstract data type class."""


class CoreDataType(StrEnum):
    """Supported core data type primitives."""

    INT = "int"
    FLOAT = "float"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class PrimitiveDataType(DataType):
    """Define core data type primitive.

    Args:
        data_type (CoreDataType):

    """

    _data_type: CoreDataType

    def __init__(self, data_type: CoreDataType) -> None:
        self._data_type = data_type

    @property
    def primitive_data_type(self) -> CoreDataType:
        """PrimitiveDataType data type."""
        return self._data_type


class TemplateDataType(DataType):
    """Define template struct type.

    Args:
        data_type (Identifier): template type identifier
        widths (list[int], optional): fixed bit size width of type

    """

    _data_type: _Identifier
    widths: list[int] | None

    def __init__(self, data_type: _Identifier, widths: list[int] | None = None) -> None:
        self._data_type = data_type
        self.widths = widths

    @property
    def template_type(self) -> _Identifier:
        """Template Identifier."""
        return self._data_type


class NumericalType(Type):
    """Vector array of a given DataType and shape.

    Args:
        data_type (DataType): Type information of data contained in vector
        shape (List[Expression]): Shape of vector

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
        types (List[Type]): types of each element within the tuple

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
