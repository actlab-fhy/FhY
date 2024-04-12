import pytest
from antlr4 import InputStream, CommonTokenStream, RecognitionException, BailErrorStrategy
from antlr4.error.ErrorListener import ErrorListener
from fhy.lang.ast import Argument, ASTNode, Component, Module, Procedure, QualifiedType
from fhy.lang.ast_builder import from_parse_tree
from fhy.lang.parser import FhYLexer, FhYParser
from fhy.ir import DataType, PrimitiveDataType, NumericalType, TypeQualifier


class ThrowingErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise SyntaxError(f"Syntax error at {line}:{column} - {msg}")


@pytest.fixture(scope="module")
def lexer():
    def create_lexer(input_str):
        input_stream = InputStream(input_str)
        lexer = FhYLexer(input_stream)
        lexer.removeErrorListeners()
        lexer.addErrorListener(ThrowingErrorListener())
        return lexer
    return create_lexer


@pytest.fixture(scope="module")
def parser(lexer):
    def create_parser(input_str):
        lexer_instance = lexer(input_str)
        token_stream = CommonTokenStream(lexer_instance)
        parser = FhYParser(token_stream)
        parser._errHandler = BailErrorStrategy()
        parser.removeErrorListeners()
        parser.addErrorListener(ThrowingErrorListener())
        return parser
    return create_parser


def test_empty_file(parser):
    """Test that an empty file is converted correctly."""
    source_file_content = ""
    parse_tree = parser(source_file_content).module()
    assert parse_tree is not None

    ast: ASTNode = from_parse_tree(parse_tree)

    assert isinstance(ast, Module), "Expected Module AST node"
    assert len(ast.components) == 0, "Expected empty module"


def test_empty_procedure(parser):
    """Test that an empty procedure is converted correctly."""
    source_file_content = "proc foo(){}"
    parse_tree = parser(source_file_content).module()
    assert parse_tree is not None

    ast: ASTNode = from_parse_tree(parse_tree)

    assert isinstance(ast, Module), "Expected Module AST node"
    assert len(ast.components) == 1, "Expected 1 component"
    procedure: Component = ast.components[0]
    assert isinstance(procedure, Procedure), "Expected Procedure AST node"
    assert procedure.name.name_hint == "foo", "Expected procedure name to be 'foo'"
    assert len(procedure.args) == 0, "Expected 0 arguments"
    assert len(procedure.body) == 0, "Expected 0 statements in the body"


def test_empty_procedure_with_a_qualified_argument(parser):
    """Test that an empty procedure with a single qualified argument is converted correctly."""
    source_file_content = "proc foo(input int32 x){}"
    parse_tree = parser(source_file_content).module()
    assert parse_tree is not None

    ast: ASTNode = from_parse_tree(parse_tree)

    assert isinstance(ast, Module), "Expected Module AST node"
    assert len(ast.components) == 1, "Expected 1 component"
    procedure: Component = ast.components[0]
    assert isinstance(procedure, Procedure), "Expected Procedure AST node"
    assert procedure.name.name_hint == "foo", "Expected procedure name to be 'foo'"
    assert len(procedure.args) == 1, "Expected 1 argument"
    arg: Argument = procedure.args[0]
    assert arg.name.name_hint == "x", "Expected argument name to be 'x'"
    arg_type: QualifiedType = arg.qualified_type
    assert arg_type.type_qualifier == TypeQualifier.INPUT, "Expected argument type qualifier to be INPUT"
    assert isinstance(arg_type.base_type, NumericalType), "Expected argument data type to be numerical"
    arg_base_type: NumericalType = arg_type.base_type
    assert len(arg_base_type.shape) == 0, "Expected argument data type to have an empty shape (i.e., scalar)"
    arg_base_data_type: DataType = arg_base_type.data_type
    assert arg_base_data_type.primitive_data_type == PrimitiveDataType.INT32, "Expected argument primitive data type to be INT"
    assert len(procedure.body) == 0, "Expected 0 statements in the body"
