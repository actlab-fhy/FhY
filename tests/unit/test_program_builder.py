"""Test Driver Program Builder and Peripherals."""

import os
from pathlib import Path
from typing import List, Tuple

import pytest

from fhy.driver import utils
from fhy.driver.ast_program_builder.builder import ASTProgramBuilder
from fhy.driver.ast_program_builder.module_tree import ModuleTree
from fhy.driver.ast_program_builder.source_file_ast import SourceFileAST
from fhy.driver.compilation_options import CompilationOptions
from fhy.driver.workspace import Workspace


@pytest.fixture
def data_dir() -> str:
    path = os.path.join(__file__, os.pardir, "data")

    return os.path.abspath(path)


def _workspace(_path: str) -> Workspace:
    return Workspace(root=Path(_path))


@pytest.fixture
def circular_dir(data_dir) -> Workspace:
    return _workspace(os.path.join(data_dir, "circular_import", "a.fhy"))


@pytest.fixture
def unidirectional_import(data_dir) -> Workspace:
    return _workspace(os.path.join(data_dir, "unidirectional_import", "a.fhy"))


@pytest.fixture
def config() -> CompilationOptions:
    return CompilationOptions()


@pytest.mark.parametrize(["symbol", "expected"], [("a.b.c", (["a", "b"], "c"))])
def test_separation(symbol: str, expected: Tuple[List[str], str]):
    result_a, result_b = utils.get_imported_symbol_module_components_and_name(symbol)

    assert result_a == expected[0], "Unexpected Import Components"
    assert result_b == expected[1], "Unexpected Import Name"


def test_builder_file_asts(unidirectional_import, config):
    program = ASTProgramBuilder(unidirectional_import, config)
    ast_files = program._build_source_file_asts()

    assert isinstance(ast_files, list), "Expected to return a list"
    assert len(ast_files) == 2, f"Expected to detect 2 Files: {len(ast_files)}"
    assert all(
        isinstance(i, SourceFileAST) for i in ast_files
    ), "Expected SourceFileAST Objects"

    assert ast_files[0].path.name == "a.fhy"
    assert ast_files[1].path.name == "b.fhy"


def test_get_filepath_names(unidirectional_import, config):
    program = ASTProgramBuilder(unidirectional_import, config)
    ast_files = program._build_source_file_asts()
    paths = {i.path for i in ast_files}

    result = {program._get_module_name_from_source_file_path(j) for j in paths}
    assert result == {"a", "b"}, "Expected Source Path names to match."


def test_builder_module_tree(unidirectional_import, config):
    program = ASTProgramBuilder(unidirectional_import, config)
    ast_files = program._build_source_file_asts()
    paths = {i.path for i in ast_files}

    result = program._build_module_tree(paths)
    assert isinstance(result, ModuleTree), "Expected to return ModuleTree Object"
    assert len(result.children) == 2, "Expected Two Children"

    for child in result.children:
        assert isinstance(
            child, ModuleTree
        ), "Expected to Children to be ModuleTree Objects"
        assert len(child.children) == 0, "Expected children to be end leaf nodes."

    names = {i.name for i in result.children}
    assert names == {"root.a", "root.b"}, "Expected ModuleTree Names to be identical."
