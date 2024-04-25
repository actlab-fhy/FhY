import pytest
from fhy.lang import ast
import fhy.lang.ast
from fhy.lang.passes import build_symbol_table
from fhy.lang.passes.symbol_table_builder import UndeclaredIdentifierException, AlreadyDeclaredIdentifierException
from fhy import ir
from ..utils import construct_ast, lexer, list_to_types, parser
from fhy.lang.printer import pprint_ast
import fhy.lang



def _get_symbol_table_string_keys(symbol_table: ir.Table[ir.Identifier, ir.SymbolTableFrame]) -> set[str]:
    return {symbol.name_hint for symbol in symbol_table.keys()}


def test_empty_program(parser):
    source_file_content = ""
    _ast = construct_ast(parser, source_file_content)

    symbol_table = build_symbol_table(_ast)

    assert len(symbol_table) == 1
    module_namespace = list(symbol_table.values())[0]
    assert len(module_namespace) == 0


def test_empty_procedure(parser):
    source_file_content = "proc main() {}"
    _ast = construct_ast(parser, source_file_content)

    symbol_table = build_symbol_table(_ast)

    assert len(symbol_table) == 2

    module_namespace_symbol_table = list(symbol_table.values())[0]
    assert len(module_namespace_symbol_table) == 1
    assert "main" in _get_symbol_table_string_keys(module_namespace_symbol_table)

    main_namespace_symbol_table = list(symbol_table.values())[1]
    assert len(main_namespace_symbol_table) == 0


def test_procedure_with_arguments(parser):
    source_file_content = "proc main(input int32[A, B] a, input int32[A, C] b) {}"
    _ast = construct_ast(parser, source_file_content)

    symbol_table = build_symbol_table(_ast)

    assert len(symbol_table) == 2

    module_namespace_symbol_table = list(symbol_table.values())[0]
    assert len(module_namespace_symbol_table) == 1
    assert "main" in _get_symbol_table_string_keys(module_namespace_symbol_table)

    main_namespace_symbol_table = list(symbol_table.values())[1]
    assert len(main_namespace_symbol_table) == 5
    assert "a" in _get_symbol_table_string_keys(main_namespace_symbol_table)
    assert "b" in _get_symbol_table_string_keys(main_namespace_symbol_table)
    assert "A" in _get_symbol_table_string_keys(main_namespace_symbol_table)
    assert "B" in _get_symbol_table_string_keys(main_namespace_symbol_table)
    assert "C" in _get_symbol_table_string_keys(main_namespace_symbol_table)


def test_procedure_with_declaration_statement(parser):
    source_file_content = "proc main(input int32[A, B] a) {temp int32[A] b;}"
    _ast = construct_ast(parser, source_file_content)

    symbol_table = build_symbol_table(_ast)

    assert len(symbol_table) == 2

    module_namespace_symbol_table = list(symbol_table.values())[0]
    assert len(module_namespace_symbol_table) == 1
    assert "main" in _get_symbol_table_string_keys(module_namespace_symbol_table)

    main_namespace_symbol_table = list(symbol_table.values())[1]
    assert len(main_namespace_symbol_table) == 4
    assert "a" in _get_symbol_table_string_keys(main_namespace_symbol_table)
    assert "b" in _get_symbol_table_string_keys(main_namespace_symbol_table)
    assert "A" in _get_symbol_table_string_keys(main_namespace_symbol_table)
    assert "B" in _get_symbol_table_string_keys(main_namespace_symbol_table)


def test_fails_with_undefined_shape_variable(parser):
    source_file_content = "proc main(input int32[A, B] a) {temp int32[C] b;}"
    _ast = construct_ast(parser, source_file_content)

    with pytest.raises(UndeclaredIdentifierException):
        build_symbol_table(_ast)


def test_fails_with_already_defined_variable(parser):
    source_file_content = "proc main(input int32[A, B] a) {temp int32[A] a;}"
    _ast = construct_ast(parser, source_file_content)

    with pytest.raises(AlreadyDeclaredIdentifierException):
        build_symbol_table(_ast)


def test_fails_with_already_defined_procedure(parser):
    source_file_content = "proc main() {} proc main() {}"
    _ast = construct_ast(parser, source_file_content)

    with pytest.raises(AlreadyDeclaredIdentifierException):
        build_symbol_table(_ast)


def test_import_variable(parser):
    source_file_content = "import constants.pi; proc main() {temp int32 a = constants.pi;}"
    _ast = construct_ast(parser, source_file_content)

    symbol_table = build_symbol_table(_ast)

    assert len(symbol_table) == 2

    module_namespace_symbol_table = list(symbol_table.values())[0]
    assert len(module_namespace_symbol_table) == 2
    assert "main" in _get_symbol_table_string_keys(module_namespace_symbol_table)
    assert "constants.pi" in _get_symbol_table_string_keys(module_namespace_symbol_table)

    main_namespace_symbol_table = list(symbol_table.values())[1]
    assert len(main_namespace_symbol_table) == 1
    assert "a" in _get_symbol_table_string_keys(main_namespace_symbol_table)


def test_fails_with_already_defined_import(parser):
    source_file_content = "import constants.pi; import constants.pi;"
    _ast = construct_ast(parser, source_file_content)

    with pytest.raises(AlreadyDeclaredIdentifierException):
        build_symbol_table(_ast)
