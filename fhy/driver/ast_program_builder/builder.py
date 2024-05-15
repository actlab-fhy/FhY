"""Fhy Builder Module."""

import logging
from collections import deque
from pathlib import Path
from typing import List, Optional, Set

import networkx as nx

from fhy import ir
from fhy.lang import collect_imported_identifiers, replace_identifiers
from fhy.utils import error

from ..compilation_options import CompilationOptions
from ..utils import get_imported_symbol_module_components_and_name
from ..workspace import Workspace
from .module_tree import ModuleTree
from .source_file_ast import SourceFileAST, build_source_file_ast

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class ASTProgramBuilder(object):
    """Construct an AST Program.

    Args:
        workspace (Workspace): describe project root directory file.
        options (CompilationOptions): Configuration options during compilation

    Raises:
        FhYImportError: Problematic Import statement detected.

    Returns:
        _type_: _description_

    """

    _workspace: Workspace
    _options: CompilationOptions

    def __init__(self, workspace: Workspace, options: CompilationOptions):
        self._workspace = workspace
        self._options = options

    @property
    def root_dir(self) -> Path:
        return self._workspace.root

    @property
    def src_dir(self):
        return self._workspace.source

    def build(self) -> ir.Program:
        unresolved_source_file_asts = self._build_source_file_asts()
        paths: Set[Path] = {i.path for i in unresolved_source_file_asts}
        module_tree: ModuleTree = self._build_module_tree(paths)
        resolved_source_file_asts = self._resolve_imports(
            unresolved_source_file_asts, module_tree
        )
        ast_program = self._build_program(resolved_source_file_asts)

        return ast_program

    def _build_source_file_asts(self) -> List[SourceFileAST]:
        """This Compiles Files Propagating Outward from Main Root Module.

        Note:
            This appears to avoid any modules which are not imported, or in some
            way connected directly, or indirectly via imports. This may or may
            not be desired for end user. We will need to thoroughly document
            this expected behavior of compilation (i.e. that lone modules are
            not included in compilation)

        """
        source_file_asts: List[SourceFileAST] = []
        source_file_queue = deque([self._workspace.main])

        while source_file_queue:
            filepath: Path = source_file_queue.popleft()
            paths = map(lambda k: k.path, source_file_asts)
            if any(i == filepath for i in paths):
                continue

            ast_source: SourceFileAST = build_source_file_ast(filepath)
            source_file_asts.append(ast_source)

            imported_identifiers: Set[ir.Identifier] = collect_imported_identifiers(
                ast_source.ast
            )
            imported_symbols: Set[str] = {
                import_identifier.name_hint
                for import_identifier in imported_identifiers
            }
            import_paths: Set[Path] = set(
                self._get_source_file_path_from_imported_symbol(imported_symbol)
                for imported_symbol in imported_symbols
            )
            source_file_queue.extend(import_paths)

        return source_file_asts

    def _build_module_tree(self, filepaths: Set[Path]) -> ModuleTree:
        """Construct a Module Tree Node Graph from a set of Filepaths.

        Args:
            filepaths (Set[Path]): Unique set of connected Filepaths.

        Returns:
            ModuleTree: Interconnected Module Tree

        """
        tree = ModuleTree("root")
        module_names: Set[str] = {
            self._get_module_name_from_source_file_path(path) for path in filepaths
        }

        for module_name in module_names:
            current_tree: ModuleTree = tree
            for source_file_name in module_name.split("."):
                options = {child.module_name for child in current_tree.children}
                if source_file_name not in options:
                    new_node = ModuleTree(source_file_name, parent=current_tree)
                    current_tree.children.add(new_node)
                    current_tree = new_node
                else:
                    current_tree = next(
                        child
                        for child in current_tree.children
                        if child.module_name == source_file_name
                    )

        return tree

    def _get_source_file_path_from_imported_symbol(
        self,
        imported_symbol: str,
    ) -> Path:
        """Construct a Filepath Representation from Import Name Hint Symbol.

        Args:
            imported_symbol (str): Name Hint Symbol from Import Statement Identifier

        Returns:
            Path: Module Path to a given import symbol.

        """
        route, name = get_imported_symbol_module_components_and_name(imported_symbol)
        import_path = Path(*route, name).with_suffix(".fhy")

        return self.root_dir / import_path

    def _get_module_name_from_source_file_path(self, filepath: Path) -> str:
        path: str = str(filepath.relative_to(self.root_dir).with_suffix(""))

        return path.replace("/", ".")

    def _confirm_import_exists(self, identifier: ir.Identifier, tree: ModuleTree):
        # TODO: Implement Confirmation Check.
        ...

    def _is_cyclical(self, graph: nx.Graph) -> Optional[list]:
        try:
            result = list(nx.find_cycle(graph, orientation="ignore"))
            return result

        except nx.NetworkXNoCycle:
            ...

    def _get_module_by_name(self, tree: ModuleTree, name: str) -> Optional[ModuleTree]:
        for a in name.split("."):
            tree = next(i for i in tree.children if i.module_name == a)
            if tree is None:
                break

        return tree

    def _get_module_by_path(self, tree: ModuleTree, path: Path) -> Optional[ModuleTree]:
        route = self._get_module_name_from_source_file_path(path)

        return self._get_module_by_name(tree, route)

    def _resolve_imports(
        self, source_file_asts: List[SourceFileAST], module_tree: ModuleTree
    ) -> List[SourceFileAST]:
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
                relevant_module = self._get_module_by_name(module_tree, iid.name_hint)

                if relevant_module is None:
                    msg = f"Invalid Import Statement. Module Not Found: {iid}"
                    log.error(msg)
                    raise error.FhYImportError(msg)

                validate = self._confirm_import_exists(iid, relevant_module)
                if not validate:
                    msg = f"Import Not Found: {iid}"
                    log.error(msg)
                    raise error.FhYImportError(msg)

                # graph.add_edge(source.path.name, leaf)

        # Cycle Detection
        if result := self._is_cyclical(graph):
            msg = f"Circular Import Detected: {result}"
            log.error(msg)
            raise error.FhYImportError(msg)

        # Resolve Import Identifiers
        id_map = {}  # TODO: Collect Map Somehow
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
