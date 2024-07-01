"""Bitwise f-DFG operations."""
from ..op import Op
from fhy.ir.identifier import Identifier
from fhy.ir.type import NumericalType, DataType, PrimitiveDataType


bitwise_not_op = Op(Identifier("bitwise_not"))
bitwise_and_op = Op(Identifier("bitwise_and"))
bitwise_or_op = Op(Identifier("bitwise_or"))
bitwise_xor_op = Op(Identifier("bitwise_xor"))
left_shift_op = Op(Identifier("left_shift"))
right_shift_op = Op(Identifier("right_shift"))
