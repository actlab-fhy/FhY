"""Main Compilation Driver."""

from collections import deque
from pathlib import Path
from typing import Dict

from fhy import ir
from fhy.lang import ast, collect_imported_identifiers, from_fhy_source
from fhy.lang.pprint import pformat_ast

from .ast_program_builder import build_ast_program
from .ast_program_builder.utils import get_import_modules_and_name
from .compilation_options import CompilationOptions
from .file_reader import read_file
from .workspace import Workspace


def _resolve_file_path_from_import_name(import_name: str, root_path: Path) -> Path:
    root_path_directory = root_path.parent
    import_module_list, _ = get_import_modules_and_name(import_name)
    import_path = Path(*import_module_list).with_suffix(".fhy")

    return root_path_directory.joinpath(import_path)


def _convert_source_files_to_ASTs(workspace: Workspace) -> Dict[Path, ast.Module]:
    root_path = workspace.root
    source_file_queue = deque([root_path])

    ast_map: Dict[Path, ast.Module] = {}
    while source_file_queue:
        source_file_path = source_file_queue.popleft()

        if source_file_path in ast_map:
            continue

        source_text = read_file(source_file_path)
        source_ast = from_fhy_source(source_text)
        pformat_ast(source_ast, is_identifier_id_printed=True)
        ast_map[source_file_path] = source_ast

        import_identifiers = collect_imported_identifiers(source_ast)
        import_names = set(
            import_identifier.name_hint for import_identifier in import_identifiers
        )
        import_paths = set(
            _resolve_file_path_from_import_name(import_name, source_file_path)
            for import_name in import_names
        )
        source_file_queue.extend(import_paths)

    return ast_map


def compile_fhy(workspace: Workspace, options: CompilationOptions) -> ir.Program:
    """Compile Fhy Source into a Program."""
    ast_program = build_ast_program(workspace, options)

    return ast_program
