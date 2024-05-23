"""Common Utilities (not fixtures) used in Unit and Integration Tests."""

import json
from typing import Any, List

from fhy.lang.ast.serialization.to_json import AlmostJson, to_almost_json


def list_to_types(xs: List[Any]) -> List[type]:
    """Convert a List of objects into a list of object types."""
    return [type(x) for x in xs]


def load_text(text: str) -> AlmostJson:
    """Utility Function for this module to convert json text into AlmostJson Object."""
    return json.loads(text, object_hook=to_almost_json)
