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
class PrimitiveDataType(StrEnum):
    """Core Supported Primitive Types"""
    INT32 = "int32"
    FLOAT32 = "float32"


class DataType:
    """Data Type Defines Core Type Primitive, but of Flexible Bit Width.

    Args:
        _primitive_type: (PrimitiveType):

    """
    _primitive_data_type: PrimitiveDataType

    def __init__(self,
                 _primitive_data_type: PrimitiveDataType,
                 ) -> None:
        self._primitive_data_type = _primitive_data_type

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

    def __init__(self,
                 _data_type: DataType,
                 _shape: List[Expression]
                 ) -> None:
        super().__init__()
        self._data_type = _data_type
        self._shape = _shape

    @property
    def data_type(self) -> DataType:
        return self._data_type

    @property
    def shape(self) -> List[Expression]:
        return self._shape

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


class QualifiedType(ASTNode):
    """Qualified Type Container

    Args:
        _base_type (Type): Primitive or Generic Type (e.g. float, int, T)
        _type_qualifier (Optional[TypeQualifier]): Qualifying Type, (i.e. input, output, param, state)

    e.g. Return Types are not assigned a proper name

    """
    _base_type: Type
    _type_qualifier: Optional[TypeQualifier]

    def __init__(self,
                 base_type: Type,
                 type_qualifier: Optional[TypeQualifier] = None
                 ) -> None:
        super().__init__()
        self._base_type = base_type
        self._type_qualifier = type_qualifier

    @property
    def base_type(self) -> Type:
        return self._base_type

    @property
    def type_qualifier(self) -> Optional[TypeQualifier]:
        return self._type_qualifier

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_base_type", "_type_qualifier"])
        return attrs
