"""
test_ast.py

"""

import pytest

from fhy.lang.ast import core


@pytest.mark.parametrize("name", ["test", "nombre", "badHombre", "Example"])
def test_base_node_name(name):
    """Confirm that the Given Class Name is Carried through Inheritance."""
    # Dynamically Construct a Class with a given Name
    obj = type(name, (core.ASTNode,), {})
    ret = obj.keyname()
    assert (
        ret == name
    ), f"Names are Not Equivalent: Returned(`{ret}`) vs Given(`{name}`)"
