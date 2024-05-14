"""Fhy Builder Module."""

import logging
from collections import deque
from pathlib import Path
from typing import List, Optional, Set, Tuple

import networkx as nx

from fhy import ir
from fhy.lang import collect_imported_identifiers, replace_identifiers
from fhy.utils import error

from ..compilation_options import CompilationOptions
from ..workspace import Workspace
from .module_tree import ModuleTree
from .source_file_ast import SourceFileAST, build_source_file_ast

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def _get_imported_symbol_module_components_and_name(
    imported_symbol: str,
) -> Tuple[List[str], str]:
    import_components = imported_symbol.split(".")
    import_module_components = import_components[:-1]
    imported_name = import_components[-1]

    return import_module_components, imported_name


class ASTProgramBuilder(object):
    """_summary_.

    Args:
        object (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_

    """

    _workspace: Workspace
    _options: CompilationOptions

    def __init__(self, workspace: Workspace, options: CompilationOptions):
        self._workspace = workspace
        self._options = options

    def build(self) -> ir.Program:
        unresolved_source_file_asts = self._build_source_file_asts()
        module_tree = self._build_module_tree(unresolved_source_file_asts)
        resolved_source_file_asts = self._resolve_imports(
            unresolved_source_file_asts, module_tree
        )
        ast_program = self._build_program(resolved_source_file_asts)

        return ast_program

    def _build_source_file_asts(self) -> Set[SourceFileAST]:
        source_file_asts = set()
        source_file_queue = deque([self._workspace.root])

        while source_file_queue:
            source_file_path = source_file_queue.popleft()
            if source_file_path in source_file_asts:
                continue

            source_file_ast = build_source_file_ast(source_file_path)
            source_file_asts.add(source_file_ast)

            imported_identifiers = collect_imported_identifiers(source_file_ast.ast)
            imported_symbols = {
                import_identifier.name_hint
                for import_identifier in imported_identifiers
            }
            import_paths = set(
                self._get_source_file_path_from_imported_symbol(imported_symbol)
                for imported_symbol in imported_symbols
            )
            source_file_queue.extend(import_paths)

        return source_file_asts

    def _build_module_tree(self, source_file_paths: Set[Path]) -> ModuleTree:
        tree = ModuleTree("root")
        module_names = {
            self._get_module_name_from_source_file_path(path)
            for path in source_file_paths
        }

        for module_name in module_names:
            current_tree = tree
            for source_file_name in module_name.split("."):
                if source_file_name not in {
                    child.name for child in current_tree.children
                }:
                    new_node = ModuleTree(source_file_name, parent=current_tree)
                    current_tree.children.add(new_node)
                current_tree = next(
                    child
                    for child in current_tree.children
                    if child.name == source_file_name
                )

        return tree

    def _get_source_file_path_from_imported_symbol(
        self,
        imported_symbol: str,
    ) -> Path:
        root_path_directory = self._workspace.root.parent
        import_module_list, _ = _get_imported_symbol_module_components_and_name(
            imported_symbol
        )
        import_path = Path("/".join(import_module_list)).with_suffix(".fhy")

        return root_path_directory / import_path

    def _get_module_name_from_source_file_path(self, source_file_path: Path) -> str:
        root_directory_path = self._workspace.root.parent

        return str(
            source_file_path.relative_to(root_directory_path).with_suffix("")
        ).replace("/", ".")

    def _confirm_import_exists(self, identifier: ir.Identifier, tree: ModuleTree):
        # TODO: Implement Confirmation Check.
        ...

    def _is_cyclical(self, graph: nx.Graph) -> Optional[list]:
        try:
            result = list(nx.find_cycle(graph, orientation="ignore"))
            return result

        except nx.NetworkXNoCycle:
            ...

    def _resolve_imports(
        self, source_file_asts: Set[SourceFileAST], module_tree: ModuleTree
    ) -> Set[SourceFileAST]:
        # TODO: fill in;
        #   1. iterate over all of the imported symbols (i.e., iterate over all modules,
        #      use the get imported identifier pass to grab, and iterate over those
        #      imported identifiers), use the module tree to check that the symbol
        #      exists in the file, and create a networkx graph of the dependencies
        #      between the modules.
        #   2. check for cycles on the networkx graph using built-in networkx functions
        #      (raise an exception if there is a cycle)
        #   3. iterate over the imported symbols and replace the identifier in the
        #      module where the identifier was imported with the original identifier (I
        #      wrote a pass that replaces identifiers; use that)
        #   4. return the set of source file asts with the imports resolved

        graph = nx.Graph()

        for source in source_file_asts:
            import_ids: Set[ir.Identifier] = collect_imported_identifiers(source.ast)
            for iid in import_ids:
                if not (leaf := self._confirm_import_exists(iid, module_tree)):
                    msg = f"Import Not Found: {iid}"
                    log.error(msg)
                    raise error.FhYImportError(msg)

                graph.add_edge(source.path, leaf)

        # Cycle Detection
        if result := self._is_cyclical(graph):
            msg = f"Circular Import Detected: {result}"
            log.error(msg)
            raise error.FhYImportError(msg)

        # Resolve Import Identifiers
        id_map = {}
        bank = set()
        for k in source_file_asts:
            bank.add(replace_identifiers(k.ast, id_map))

        return bank

    def _build_program(self, source_file_asts: Set[SourceFileAST]) -> ir.Program:
        # TODO: Chris will change this later
        program = ir.Program()
        for source_file_ast in source_file_asts:
            program._components[source_file_ast.ast.name] = source_file_ast.ast

        return program


def build_ast_program(workspace: Workspace, options: CompilationOptions) -> ir.Program:
    """Build an AST Program.

    Args:
        workspace (Workspace): _description_
        options (CompilationOptions): _description_

    Returns:
        ir.Program: _description_

    """
    builder = ASTProgramBuilder(workspace, options)

    return builder.build()
