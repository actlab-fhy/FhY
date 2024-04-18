""" """
import pytest

from fhy.ir.type import PrimitiveDataType, TypeQualifier
from fhy.lang.ast_builder import builder


@pytest.mark.parametrize(
    "value, obj, expect",
    [
        ("int32", PrimitiveDataType, PrimitiveDataType.INT32),
        ("float32", PrimitiveDataType, PrimitiveDataType.FLOAT32),
        pytest.param(
            "InT32",
            PrimitiveDataType,
            PrimitiveDataType.INT32,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            "_PLACEHOLDER",
            PrimitiveDataType,
            PrimitiveDataType._PLACEHOLDER,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            "_test", PrimitiveDataType, None, marks=pytest.mark.xfail(raises=ValueError)
        ),
        ("input", TypeQualifier, TypeQualifier.INPUT),
        ("output", TypeQualifier, TypeQualifier.OUTPUT),
        ("param", TypeQualifier, TypeQualifier.PARAM),
        ("temp", TypeQualifier, TypeQualifier.TEMP),
        pytest.param(
            "InPuT", TypeQualifier, None, marks=pytest.mark.xfail(raises=ValueError)
        ),
        pytest.param(
            "test", TypeQualifier, None, marks=pytest.mark.xfail(raises=ValueError)
        ),
    ],
)
def test_validation(value, obj, expect):
    result = builder.validate(value, obj)
    assert isinstance(
        result, obj
    ), f"Return Value Must be an Instance of StrEnum: {result}"
    assert result == expect, "Did not Return Expected Result: {result} vs {expect}"
