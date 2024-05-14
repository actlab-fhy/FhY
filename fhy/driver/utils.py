from typing import List, Tuple


def get_imported_symbol_module_components_and_name(
    imported_symbol: str,
) -> Tuple[List[str], str]:
    import_components = imported_symbol.split(".")
    import_module_components = import_components[:-1]
    imported_name = import_components[-1]

    return import_module_components, imported_name
