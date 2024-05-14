"""Test Driver Program Builder and Peripherals."""

from typing import List, Tuple

import pytest

from fhy.driver import utils


@pytest.mark.parametrize(["symbol", "expected"], [("a.b.c", (["a", "b"], "c"))])
def test_separation(symbol: str, expected: Tuple[List[str], str]):
    result_a, result_b = utils.get_imported_symbol_module_components_and_name(symbol)

    assert result_a == expected[0], "Unexpected Import Components"
    assert result_b == expected[1], "Unexpected Import Name"
