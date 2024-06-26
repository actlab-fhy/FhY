# Copyright (c) 2024 FhY Developers
# Christopher Priebe <cpriebe@ucsd.edu>
# Jason C Del Rio <j3delrio@ucsd.edu>
# Hadi S Esmaeilzadeh <hadi@ucsd.edu>
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be
# used to endorse or promote products derived from this software without specific prior
# written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""FhY builder module.

Orchestrate the build of a FhY source into an ir.Program.

"""

import logging
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx  # type: ignore[import-untyped]

from fhy import error, ir
from fhy.lang import collect_imported_identifiers
from fhy.lang.ast.passes import build_symbol_table, replace_identifiers

from ..compilation_options import CompilationOptions
from ..utils import get_imported_symbol_module_components_and_name
from ..workspace import Workspace
from .module_tree import ModuleTree
from .source_file_ast import SourceFileAST, build_source_file_ast

_log: logging.Logger = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


def _get_relative_path(path: Path, ref: Path) -> str:
    relative: Path = path.relative_to(ref).with_suffix("")

    return ".".join(relative.parts)


class ASTProgramBuilder(object):
    """Construct an AST ir.Program.

    Args:
        workspace (Workspace): Define project root directory file.
        options (CompilationOptions): Configuration options during compilation
        log (logging.Logger): Define a logger to report for debugging purposes.

    Raises:
        FhYImportError: Problematic Import statement detected.

    """

    _workspace: Workspace
    _options: CompilationOptions
    log: logging.Logger

    def __init__(
        self,
        workspace: Workspace,
        options: CompilationOptions,
        log: logging.Logger = _log,
    ):
        self._workspace = workspace
        self._options = options
        self.log = log

    @property
    def root_dir(self) -> Path:
        """Project root directory."""
        return self._workspace.root

    @property
    def src_dir(self):
        """Source code directory."""
        return self._workspace.source

    def build(self) -> ir.Program:
        """Build an ir.Program composed of (multiple) modules."""
        unresolved_asts: List[SourceFileAST] = self._build_source_file_asts()
        paths: Set[Path] = {i.path for i in unresolved_asts}
        module_tree: ModuleTree = self._build_module_tree(paths)
        resolved_asts: List[SourceFileAST] = self._resolve_imports(
            unresolved_asts, module_tree
        )
        ast_program: ir.Program = self._build_program(resolved_asts)

        return ast_program

    def _build_source_file_asts(self) -> List[SourceFileAST]:
        """Compile files to AST, propagating outward from primary entry module.

        Note:
            This appears to avoid any modules which are not imported, or in some
            way connected directly, or indirectly via imports. This may or may
            not be desired for end user. We will need to thoroughly document
            this expected behavior of compilation (i.e. that lone modules are
            not included in compilation).

        """
        source_file_asts: List[SourceFileAST] = []
        source_file_queue = deque([self._workspace.main])

        while source_file_queue:
            filepath: Path = source_file_queue.popleft()
            paths = map(lambda k: k.path, source_file_asts)
            if any(i == filepath for i in paths):
                continue

            self.log.debug(
                "Building AST: %s", _get_relative_path(filepath, self.src_dir)
            )
            ast_source: SourceFileAST = build_source_file_ast(filepath)
            source_file_asts.append(ast_source)

            imported_identifiers: Set[ir.Identifier] = collect_imported_identifiers(
                ast_source.ast
            )
            qualname: Set[str] = {
                import_identifier.name_hint
                for import_identifier in imported_identifiers
            }
            import_paths: Set[Path] = set(
                self._get_source_file_path_from_imported_symbol(imported_symbol)
                for imported_symbol in qualname
            )
            source_file_queue.extend(import_paths)

        return source_file_asts

    def _build_module_tree(self, filepaths: Set[Path]) -> ModuleTree:
        """Construct a ModuleTree graph from a set of filepaths.

        Args:
            filepaths (Set[Path]): Unique set of connected filepaths.

        Returns:
            ModuleTree: Directory structure graph representation of a project.

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
                    # TODO: Extract out helper function to retrieve next element in
                    #       a set, meeting specific key condition (or map).
                    current_tree = next(
                        child
                        for child in current_tree.children
                        if child.module_name == source_file_name
                    )

        return tree

    def _get_source_file_path_from_imported_symbol(
        self, imported_symbol: str, no_symbol: bool = False
    ) -> Path:
        """Construct a filepath representation from import name hint symbol.

        Args:
            imported_symbol (str): Name hint symbol from import statement identifier
            no_symbol (bool): If true, do not strip the final element in the symbol.

        Returns:
            Path: Module path to a given import symbol.

        """
        route, name = get_imported_symbol_module_components_and_name(imported_symbol)
        # TODO: This is a bit of a flimsy hack, and should be made more robust, to
        #       avoid any potential future problems that may arise from this.
        if route[0] == "root":
            route = route[1:]
        if no_symbol:
            import_path = Path(*route, name).with_suffix(".fhy")
        else:
            import_path = Path(*route).with_suffix(".fhy")

        return self.root_dir / import_path

    def _get_module_name_from_source_file_path(self, filepath: Path) -> str:
        return _get_relative_path(filepath, self.root_dir)

    def _find_source(
        self, tree: ModuleTree, ast_sources: List[SourceFileAST]
    ) -> Optional[SourceFileAST]:
        path: Path = self._get_source_file_path_from_imported_symbol(tree.name, True)
        for source in ast_sources:
            if source.path == path:
                return source

        return None

    def _is_cyclical(self, graph: nx.Graph) -> Optional[list]:
        try:
            result = list(nx.find_cycle(graph, orientation="ignore"))
            return result

        except nx.NetworkXNoCycle:
            ...

        return None

    def _get_module_by_name(self, tree: ModuleTree, name: str) -> Optional[ModuleTree]:
        route, name = get_imported_symbol_module_components_and_name(name)

        for a in route:
            tree = next(i for i in tree.children if i.module_name == a)
            if tree is None:
                break

        return tree

    def _get_module_by_path(self, tree: ModuleTree, path: Path) -> Optional[ModuleTree]:
        route = self._get_module_name_from_source_file_path(path)

        return self._get_module_by_name(tree, route)

    def _confirm_import_exists(
        self, name_hint: str, reference_table: ir.Table
    ) -> Optional[ir.Identifier]:
        for symbol in reference_table.keys():
            if symbol.name_hint == name_hint:
                return symbol

        return None

    def _resolve_imports(
        self, source_file_asts: List[SourceFileAST], module_tree: ModuleTree
    ) -> List[SourceFileAST]:
        # TODO: Refactor to extract out into smaller more testable pieces.
        # NOTE: We currently only support a specific import style.
        #       We cannot import modules, but rather symbols from modules.
        #       All imports are absolute paths, not relative.
        #       Do not define yet a method to import outside of package (e.g. stdlib)
        #       Aliasing not yet supported (e.g. \"import x as y;\").
        # Initialize a Directional Graph
        graph = nx.DiGraph()
        resolved_sources: List[SourceFileAST] = []

        for source in source_file_asts:
            _rel_path = self._get_module_name_from_source_file_path(source.path)
            self.log.debug("Resolving Imports: %s", _rel_path)

            id_map: Dict[ir.Identifier, ir.Identifier] = {}
            import_ids: Set[ir.Identifier] = collect_imported_identifiers(source.ast)
            for iid in import_ids:
                relevant_module = self._get_module_by_name(module_tree, iid.name_hint)

                if relevant_module is None:
                    msg = f"Invalid Import Statement. Module Not Found: {iid}"
                    self.log.error(msg)
                    raise error.FhYImportError(msg)

                # Find Source of Import
                source_imported: Optional[SourceFileAST] = self._find_source(
                    relevant_module, source_file_asts
                )

                if source_imported is None:
                    raise Exception("Make a Better Module not Found Error.")

                # Build symbol tables to properly handle identifier scope
                table_from: ir.SymbolTable = build_symbol_table(source_imported.ast)
                module_context: ir.Table = list(table_from.values())[0]

                # Confirm Identifiers derived from use of given Import within source
                _, name = get_imported_symbol_module_components_and_name(iid.name_hint)
                exists = self._confirm_import_exists(name, module_context)
                if not exists:
                    msg = (
                        f'Import symbol "{name}" Not Found within: '
                        f"{source_imported.path}"
                    )
                    self.log.error(msg)
                    raise error.FhYImportError(msg)

                id_map[iid] = exists

                b = self._get_module_name_from_source_file_path(source_imported.path)
                graph.add_edge(_rel_path, b)

            self.log.debug("Completed Resolving Imports: %s", _rel_path)
            _ast = replace_identifiers(source.ast, id_map)
            resolved_sources.append(replace(source, ast=_ast))

        # Cycle Detection
        if result := self._is_cyclical(graph):
            msg = f"Circular Import Detected: {result}"
            self.log.error(msg)
            raise error.FhYImportError(msg)

        return resolved_sources

    def _build_program(self, source_file_asts: List[SourceFileAST]) -> ir.Program:
        # TODO: Chris will change this later
        program = ir.Program()
        for source_file_ast in source_file_asts:
            program._components[source_file_ast.ast.name] = source_file_ast.ast

        return program


def build_ast_program(
    workspace: Workspace, options: CompilationOptions, log: logging.Logger = _log
) -> ir.Program:
    """Build an ir.Program.

    Args:
        workspace (Workspace): Define project root directory file.
        options (CompilationOptions): Configuration options during compilation
        log (optional, logging.Logger): Provide a logger for debugging purposes.

    Raises:
        FhYImportError: Problematic Import statement detected.

    Returns:
        (ir.Program):

    """
    builder = ASTProgramBuilder(workspace, options, log)

    return builder.build()
