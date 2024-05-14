"""Driver Utilities."""

from typing import List, Tuple


def get_imported_symbol_module_components_and_name(
    imported_symbol: str,
) -> Tuple[List[str], str]:
    """Separate Symbol Import Name from components."""
    import_components: List[str] = imported_symbol.split(".")
    *import_module_components, imported_name = import_components

    return import_module_components, imported_name
