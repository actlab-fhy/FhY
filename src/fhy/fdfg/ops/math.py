"""Mathematical f-DFG operations."""
from ..op import Op
from fhy.ir.identifier import Identifier
from fhy.ir.type import NumericalType, DataType, PrimitiveDataType


neg_op = Op(Identifier("neg"))

add_op = Op(Identifier("add"))
sub_op = Op(Identifier("sub"))
mul_op = Op(Identifier("mul"))
div_op = Op(Identifier("div"))
floor_div_op = Op(Identifier("floor_div"))
mod_op = Op(Identifier("mod"))
pow_op = Op(Identifier("pow"))
