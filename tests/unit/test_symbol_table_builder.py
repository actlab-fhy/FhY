"""Unit Test Symbol Table Builder Module."""

import pytest
from fhy import error, ir
from fhy.lang.ast.passes import build_symbol_table


def _get_symbol_table_string_keys(
    symbol_table: ir.Table[ir.Identifier, ir.SymbolTableFrame],
) -> set[str]:
    return {symbol.name_hint for symbol in symbol_table.keys()}


def test_empty_program(construct_ast):
    """Test an empty program.

    builds a symbol table with one namespace, containing no variables.

    """
    source_file_content = ""
    _ast = construct_ast(source_file_content)

    symbol_table = build_symbol_table(_ast)

    assert len(symbol_table) == 1
    module_namespace = list(symbol_table.values())[0]
    assert len(module_namespace) == 0


def test_empty_procedure(construct_ast):
    """Tests Empty Procedure Body contains procedure name in symbol table."""
    source_file_content = "proc main() {}"
    _ast = construct_ast(source_file_content)

    symbol_table = build_symbol_table(_ast)

    assert len(symbol_table) == 2

    module_namespace_symbol_table = list(symbol_table.values())[0]
    assert len(module_namespace_symbol_table) == 1
    assert "main" in _get_symbol_table_string_keys(module_namespace_symbol_table)

    main_namespace_symbol_table = list(symbol_table.values())[1]
    assert len(main_namespace_symbol_table) == 0


def test_procedure_with_arguments(construct_ast):
    """Test Empty Procedure with Arguments.

    names are in the symbol table, but within procedure namespace.

    """
    source_file_content = "proc main(input int32[A, B] a, input int32[A, C] b) {}"
    _ast = construct_ast(source_file_content)

    symbol_table = build_symbol_table(_ast)

    assert len(symbol_table) == 2

    module_namespace_symbol_table = list(symbol_table.values())[0]
    assert len(module_namespace_symbol_table) == 1
    assert "main" in _get_symbol_table_string_keys(module_namespace_symbol_table)

    main_namespace_symbol_table = list(symbol_table.values())[1]
    assert len(main_namespace_symbol_table) == 5
    for char in "abABC":
        assert char in _get_symbol_table_string_keys(
            main_namespace_symbol_table
        ), f'Expected Variable in Symbol table: "{char}"'


def test_procedure_with_declaration_statement(construct_ast):
    """Test procedure body variables are in symbol table procedure namespace."""
    source_file_content = "proc main(input int32[A, B] a) {temp int32[A] b;}"
    _ast = construct_ast(source_file_content)

    symbol_table = build_symbol_table(_ast)

    assert len(symbol_table) == 2

    module_namespace_symbol_table = list(symbol_table.values())[0]
    assert len(module_namespace_symbol_table) == 1
    assert "main" in _get_symbol_table_string_keys(module_namespace_symbol_table)

    main_namespace_symbol_table = list(symbol_table.values())[1]
    assert len(main_namespace_symbol_table) == 4

    for char in "abAB":
        assert char in _get_symbol_table_string_keys(
            main_namespace_symbol_table
        ), f'Expected Variable in Symbol table: "{char}"'


def test_fails_with_undefined_shape_variable(construct_ast):
    """Tests an error is raised with an undeclared shape variable, 'C'."""
    source_file_content = "proc main(input int32[A, B] a) {temp int32[C] b;}"
    _ast = construct_ast(source_file_content)

    with pytest.raises(error.FhYSemanticsError):
        build_symbol_table(_ast)


def test_fails_with_already_defined_variable(construct_ast):
    """Tests redefinition of a variable raises an error."""
    source_file_content = "proc main(input int32[A, B] a) {temp int32[A] a;}"
    _ast = construct_ast(source_file_content)

    with pytest.raises(error.FhYSemanticsError):
        build_symbol_table(_ast)


def test_fails_with_already_defined_procedure(construct_ast):
    """Tests that redefining a procedure name raises an error."""
    source_file_content = "proc main() {} proc main() {}"
    _ast = construct_ast(source_file_content)

    with pytest.raises(error.FhYSemanticsError):
        build_symbol_table(_ast)


def test_import_variable(construct_ast):
    """Test import and usage of a variable.

    Variable should be present at both the module and procedure level (namespace).

    """
    source_file_content = (
        "import constants.pi; proc main() {temp int32 a = constants.pi;}"
    )
    _ast = construct_ast(source_file_content)

    symbol_table = build_symbol_table(_ast)

    assert len(symbol_table) == 2

    module_namespace_symbol_table = list(symbol_table.values())[0]
    assert len(module_namespace_symbol_table) == 2
    assert "main" in _get_symbol_table_string_keys(module_namespace_symbol_table)
    assert "constants.pi" in _get_symbol_table_string_keys(
        module_namespace_symbol_table
    )

    main_namespace_symbol_table = list(symbol_table.values())[1]
    assert len(main_namespace_symbol_table) == 1
    assert "a" in _get_symbol_table_string_keys(main_namespace_symbol_table)


def test_fails_with_already_defined_import(construct_ast):
    """Tests that reimporting the same variable raises an error."""
    source_file_content = "import constants.pi; import constants.pi;"
    _ast = construct_ast(source_file_content)

    with pytest.raises(error.FhYSemanticsError):
        build_symbol_table(_ast)
