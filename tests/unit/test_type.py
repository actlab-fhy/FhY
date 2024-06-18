import pytest
from fhy.ir.type import PrimitiveDataType, promote_primitive_data_types, TypeQualifier, promote_type_qualifiers


@pytest.mark.parametrize(
    ("primitive_data_type1", "primitive_data_type2", "expected_primitive_data_type"),
    [
        (PrimitiveDataType.UINT8, PrimitiveDataType.UINT8, PrimitiveDataType.UINT8),
        (PrimitiveDataType.UINT8, PrimitiveDataType.UINT16, PrimitiveDataType.UINT16),
        (PrimitiveDataType.UINT16, PrimitiveDataType.UINT8, PrimitiveDataType.UINT16),
        (PrimitiveDataType.INT32, PrimitiveDataType.INT64, PrimitiveDataType.INT64),
        (PrimitiveDataType.FLOAT16, PrimitiveDataType.FLOAT32, PrimitiveDataType.FLOAT32),
        (PrimitiveDataType.FLOAT64, PrimitiveDataType.FLOAT16, PrimitiveDataType.FLOAT64),
        (PrimitiveDataType.COMPLEX32, PrimitiveDataType.COMPLEX64, PrimitiveDataType.COMPLEX64),
        (PrimitiveDataType.FLOAT32, PrimitiveDataType.COMPLEX32, PrimitiveDataType.COMPLEX64),
    ]
)
def test_promote_primitive_data_type(primitive_data_type1, primitive_data_type2, expected_primitive_data_type):
    assert promote_primitive_data_types(primitive_data_type1, primitive_data_type2) == expected_primitive_data_type, f"Expected the promotion of {primitive_data_type1} and {primitive_data_type2} to be {expected_primitive_data_type}."


@pytest.mark.parametrize(
    ("type_qualifer1", "type_qualifer2", "expected_type_qualifer"),
    [
        (TypeQualifier.INPUT, TypeQualifier.INPUT, TypeQualifier.TEMP),
        (TypeQualifier.STATE, TypeQualifier.PARAM, TypeQualifier.TEMP),
        (TypeQualifier.PARAM, TypeQualifier.TEMP, TypeQualifier.TEMP),
        (TypeQualifier.PARAM, TypeQualifier.PARAM, TypeQualifier.PARAM),
    ]
)
def test_promote_type_qualifiers(type_qualifer1, type_qualifer2, expected_type_qualifer):
    assert promote_type_qualifiers(type_qualifer1, type_qualifer2) == expected_type_qualifer, f"Expected the promotion of {type_qualifer1} and {type_qualifer2} to be {expected_type_qualifer}."
    