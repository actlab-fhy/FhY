import pytest

from fhy.lang.ast import Module, visitor


@pytest.fixture(scope="module")
def base_visitor():
    class Base(visitor.BasePass):
        def visit_Module(self, node: Module) -> bool:
            return True

    return Base


def test_visitor_base(base_visitor):
    """Tests the behavior of a primitive Subclass of the visitor pattern BasePass"""
    instance = base_visitor()
    node = Module(span=None)

    assert instance.visit(node), "Did not Visit Module Node."
    assert instance(node), "Did not Visit Module Node"
    assert instance.visit_Module(node), "Did Not Visit Module Node"
