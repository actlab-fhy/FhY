"""Logical f-DFG operations."""
from ..op import Op
from fhy.ir.identifier import Identifier
from fhy.ir.type import NumericalType, DataType, PrimitiveDataType


logical_not_op = Op(Identifier("logical_not"))
logical_and_op = Op(Identifier("logical_and"))
logical_or_op = Op(Identifier("logical_or"))
less_than_op = Op(Identifier("less_than"))
less_than_or_equal_op = Op(Identifier("less_than_or_equal"))
greater_than_op = Op(Identifier("greater_than"))
greater_than_or_equal_op = Op(Identifier("greater_than_or_equal"))
equal_op = Op(Identifier("equal"))
not_equal_op = Op(Identifier("not_equal"))
