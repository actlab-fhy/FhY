"""Common Utilities (not fixtures) used in Unit and Integration Tests."""

from typing import Any, List


def list_to_types(xs: List[Any]) -> List[type]:
    """Convert a List of objects into a list of object types."""
    return [type(x) for x in xs]
