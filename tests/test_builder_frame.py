""" """

import pytest

from fhy.lang.ast.core import Module
from fhy.lang.ast_builder.core import ASTBuilderFrame, FieldAttributeError


def test_builder_frame_properties():
    """Tests the Expected Behavior of the Builder Class on Instantiation."""
    builder = ASTBuilderFrame(Module)

    # Attributes should be Populated
    assert builder._attributes == set(
        ("_span", "components")
    ), "Expected Certain Attributes"

    # Those Attributes should be Assigned
    assert hasattr(builder, "_span"), "Builder should have `_span` Attribute"
    assert hasattr(builder, "components"), "Builder should have `components` Attribute"

    # We should not be able to Assign Attributes not specified by the ASTNode
    with pytest.raises(FieldAttributeError):
        builder._bad_attribute = "Ooops"

    assert not hasattr(
        builder, "_bad_attribute"
    ), "Builder should not have `_bad_attribute`"

    # Confirm users cannot Directly Modify Protected Attributes
    with pytest.raises(FieldAttributeError):
        builder._type_info = "Modified!?"
    assert (
        builder._type_info != "Modified!?"
    ), "User should not be able to Modify Protected Attribute"

    # Update Values
    assert (
        builder.components is None
    ), "Expected components Attribute to be None at first."
    builder.update(components=[])
    assert isinstance(builder.components, list), "Expected Components to be List"

    # Building the Node
    node = builder.build()
    assert isinstance(node, Module), "Expected Result of Build to be a Module ASTNode"
    assert isinstance(node.components, list), "Expected Module.components to be a List"
