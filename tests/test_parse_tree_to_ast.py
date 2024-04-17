import pytest
from antlr4 import (
    BailErrorStrategy,
    CommonTokenStream,
    InputStream,
    RecognitionException,
)
from antlr4.error.ErrorListener import ErrorListener

from fhy.ir import DataType, Identifier, NumericalType, PrimitiveDataType, TypeQualifier, Type
from fhy.lang.ast import (
    Argument,
    ASTNode,
    Component,
    Expression,
    Module,
    Operation,
    Procedure,
    QualifiedType,
)
from fhy.lang.ast.expression import (
    BinaryExpression,
    BinaryOperation,
    FloatLiteral,
    IdentifierExpression,
    IntLiteral,
    UnaryExpression,
    UnaryOperation
)
from fhy.lang.ast.statement import (
    DeclarationStatement,
    ReturnStatement
)
from fhy.lang.ast_builder import from_parse_tree
from fhy.lang.parser import FhYLexer, FhYParser


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


def _test_qual_type(
        qarg: QualifiedType,
        qtype: TypeQualifier,
        btype: Type,
        primitive: PrimitiveDataType,
        ) -> Type:
    assert (
        qarg.type_qualifier == qtype
    ), f"Expected argument type qualifier to be {qtype}"

    arg_base_type: Type = qarg.base_type
    assert isinstance(
        arg_base_type, btype
    ), f"Expected argument data type to be {btype}"

    arg_base_data_type: DataType = arg_base_type.data_type
    assert (
        arg_base_data_type.primitive_data_type == primitive
    ), "Expected argument primitive data type to be INT"

    return arg_base_type


def _test_arg(
        arg: Argument,
        name: str,
        qtype: TypeQualifier,
        btype: Type,
        primitive: PrimitiveDataType,
        ) -> Type:
    """Tests a Single Argument for Standard Checks"""
    assert isinstance(
        arg.name, Identifier
    ), "Expected Argument Name to be an Identifier"

    assert arg.name.name_hint == name, f"Expected argument name to be '{name}'"
    arg_type: QualifiedType = arg.qualified_type
    assert isinstance(arg_type, QualifiedType), f"Expected Argument.qualified_type to be a QualifiedType: {arg_type}"

    return _test_qual_type(arg_type, qtype, btype, primitive)


def _test_shape(shape: list, expected: list):
    length = len(expected)
    assert len(shape) == length, f"Expected data type Shape to have {length} Elements"

    for sh, ex in zip(shape, expected):
        assert isinstance(
            sh, Expression
        ), "Expected Expression Type for Shape Components"

        assert isinstance(sh, (IdentifierExpression, IntLiteral)), "Expected IdentifierExpression or IntLiteral"
        assert type(sh) == type(ex), "Output Shape Type Does Not Match Expected Type"

        if isinstance(ex, IntLiteral):
            assert ex.value == sh.value, "Shape Value Does Not Match"
        elif isinstance(ex, IdentifierExpression):
            assert ex._identifier.name_hint == sh._identifier.name_hint, "Shape Name Hints Do Not Match"


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
    assert parse_tree is not None, "Expected Parse Tree in Module Context"

    ast: ASTNode = from_parse_tree(parse_tree)

    assert isinstance(ast, Module), "Expected Module AST node"
    assert len(ast.components) == 1, "Expected 1 component"
    procedure: Component = ast.components[0]
    assert isinstance(procedure, Procedure), "Expected Procedure AST node"
    assert isinstance(
        procedure.name, Identifier
    ), "Expected Procedure Name to be an Identifier"
    assert procedure.name.name_hint == "foo", "Expected procedure name to be 'foo'"
    assert len(procedure.args) == 1, "Expected 1 argument"

    arg: Argument = procedure.args[0]
    arg_base_type = _test_arg(arg, "x", TypeQualifier.INPUT, NumericalType, PrimitiveDataType.INT32)
    assert (
        len(arg_base_type.shape) == 0
    ), "Expected argument data type to have an empty shape (i.e., scalar)"

    assert len(procedure.body) == 0, "Expected 0 statements in the body"


def test_empty_procedure_with_a_qualified_argument_with_shape(parser):
    source_file_content = "proc foo(input int32[m, n] x){}"
    parse_tree = parser(source_file_content).module()
    assert parse_tree is not None, "Expected Parse Tree in Module Context"
    ast: ASTNode = from_parse_tree(parse_tree)

    assert isinstance(ast, Module), "Expected Module AST node"
    assert len(ast.components) == 1, "Expected 1 component"
    procedure: Component = ast.components[0]
    assert isinstance(procedure, Procedure), "Expected Procedure AST node"
    assert isinstance(
        procedure.name, Identifier
    ), "Expected Procedure Name to be an Identifier"
    assert procedure.name.name_hint == "foo", "Expected procedure name to be 'foo'"
    assert len(procedure.args) == 1, "Expected 1 argument"
    
    arg: Argument = procedure.args[0]
    arg_base_type = _test_arg(arg, "x", TypeQualifier.INPUT, NumericalType, PrimitiveDataType.INT32)
    assert (
        len(arg_base_type.shape) == 2
    ), "Expected argument data type Shape to have Two Elements"

    expected_arg_shape = [
        IdentifierExpression(Identifier("m")),
        IdentifierExpression(Identifier("n"))
    ]

    _test_shape(arg_base_type.shape, expected_arg_shape)
    assert len(procedure.body) == 0, "Expected 0 statements in the body"


def test_empty_operation(parser):
    """test that an Empty Operation is Converted Correctly"""
    source_file_content = "op foo(){}"
    parse_tree = parser(source_file_content).module()
    assert parse_tree is not None

    ast: ASTNode = from_parse_tree(parse_tree)
    assert isinstance(ast, Module), "Expected Module AST node"
    assert len(ast.components) == 1, "Expected 1 component"
    op: Component = ast.components[0]
    assert isinstance(op, Operation), "Expected Procedure AST node"
    assert op.name.name_hint == "foo", "Expected procedure name to be 'foo'"
    assert len(op.args) == 0, "Expected 0 arguments"
    assert len(op.body) == 0, "Expected 0 statements in the body"


def test_empty_operation_return_type(parser):
    """test that an Empty Operation with a Return Type is Converted Correctly"""
    source_file_content = "op foo(input int32[n, m] x) -> output int32[n, m] {}"
    parse_tree = parser(source_file_content).module()
    assert parse_tree is not None

    ast: ASTNode = from_parse_tree(parse_tree)

    assert isinstance(ast, Module), "Expected Module AST node"
    assert len(ast.components) == 1, "Expected 1 component"
    op: Component = ast.components[0]
    assert isinstance(op, Operation), "Expected Procedure AST node"
    assert op.name.name_hint == "foo", "Expected procedure name to be 'foo'"
    assert len(op.args) == 1, "Expected 1 argument"

    arg: Argument = op.args[0]
    arg_base_type = _test_arg(arg, "x", TypeQualifier.INPUT, NumericalType, PrimitiveDataType.INT32)

    expected_arg_shape = [
        IdentifierExpression(Identifier("n")),
        IdentifierExpression(Identifier("m"))
    ]
    _test_shape(arg_base_type.shape, expected_arg_shape)

    assert len(op.body) == 0, "Expected 0 statements in the body"
    assert isinstance(op.ret_type, QualifiedType)
    base_ret = _test_qual_type(op.ret_type, TypeQualifier.OUTPUT, NumericalType, PrimitiveDataType.INT32)
    _test_shape(base_ret.shape, expected_arg_shape)


def test_declaration_statement(parser):
    """Tests a single statement."""
    source_file_content = "proc foo(){temp int32 i;}"
    parse_tree = parser(source_file_content).module()
    assert parse_tree is not None

    ast: ASTNode = from_parse_tree(parse_tree)
    assert isinstance(ast, Module), "Expected Module AST node"
    assert len(ast.components) == 1, "Expected 1 component"

    procedure: Component = ast.components[0]
    assert isinstance(procedure, Procedure), "Expected Procedure AST node"
    assert len(procedure.body) == 1, "Expected Procedure to contain 1 statement"

    statement = procedure.body[0]
    assert isinstance(statement, DeclarationStatement), "Expected Statement to be a Declaration"
    assert isinstance(statement._variable_name, Identifier), "Expected Variable Name to be an Identifier"
    assert statement._variable_name.name_hint == "i", "Expected Variable Name Hint to be `i`"

    assert isinstance(statement._variable_type, QualifiedType), "Expected Statement._variable_type to be a QualifiedType"
    base = _test_qual_type(statement._variable_type, TypeQualifier.TEMP, NumericalType, PrimitiveDataType.INT32)
    assert (
        len(base.shape) == 0
    ), "Expected argument data type to have an empty shape (i.e., scalar)"


def test_return_statement(parser):
    source_file_content = "op foo() -> temp int32 {temp int32 i = 5; return i;}"
    parse_tree = parser(source_file_content).module()
    assert parse_tree is not None

    ast: ASTNode = from_parse_tree(parse_tree)
    assert isinstance(ast, Module), "Expected Module AST node"
    assert len(ast.components) == 1, "Expected 1 component"

    procedure: Component = ast.components[0]
    assert isinstance(procedure, Operation), "Expected Operation AST node"
    assert len(procedure.body) == 2, "Expected Procedure to contain 2 statements"

    first, statement = procedure.body
    assert isinstance(first, DeclarationStatement), "Expected Declaration Statement"
    assert first._variable_name.name_hint == "i", "Expected DecalrationStatement.name_hint == `i`"
    print("First: ", first._variable_type.base_type)

    assert isinstance(statement, ReturnStatement), "Expected Return Statement"

    # TODO: Variable has been Declared. We don't want to create a different Identifier
    #       for the Expression `i` during this process.
    # assert statement.value == "i", "Unexpected Return Statement Value"


def test_unary_expressions(parser):
    source_file_content = "op foo() -> temp int32 {temp int32 i = -5;}"
    parse_tree = parser(source_file_content).module()
    assert parse_tree is not None

    ast: ASTNode = from_parse_tree(parse_tree)
    assert isinstance(ast, Module), "Expected Module AST node"
    assert len(ast.components) == 1, "Expected 1 component"

    func: Component = ast.components[0]
    assert isinstance(func, Operation), "Expected Operation AST node"
    assert len(func.body) == 1, "Expected Procedure to contain 1 statement"

    statement = func.body[0]
    assert isinstance(statement, DeclarationStatement), "Expected Declaration Statement"
    assert isinstance(statement._variable_name, Identifier), "Statement Variable Must be an Identifier"
    assert statement._variable_name.name_hint == "i"
    _test_qual_type(statement._variable_type, TypeQualifier.TEMP, NumericalType, PrimitiveDataType.INT32)

    unary = statement._expression
    assert isinstance(unary, UnaryExpression), "Expected an UnaryExpression"
    assert unary._operation == UnaryOperation.NEGATIVE, "Expected Negative UnaryOperator Operation"
    assert isinstance(unary._expression, IntLiteral), "Expected IntLiteral Expression"
    assert unary._expression.value == 5, "Expected IntLiteral Value of int(5)"


def test_binary_expressions(parser):
    source_file_content = "op foo() -> temp float32 {temp float32 i = 5.0 * 6.0;}"
    parse_tree = parser(source_file_content).module()
    assert parse_tree is not None

    ast: ASTNode = from_parse_tree(parse_tree)
    assert isinstance(ast, Module), "Expected Module AST node"
    assert len(ast.components) == 1, "Expected 1 component"

    func: Component = ast.components[0]
    assert isinstance(func, Operation), "Expected Operation AST node"
    assert len(func.body) == 1, "Expected Procedure to contain 1 statement"

    statement = func.body[0]
    assert isinstance(statement, DeclarationStatement), "Expected Declaration Statement"
    assert isinstance(statement._variable_name, Identifier), "Statement Variable Must be an Identifier"
    assert statement._variable_name.name_hint == "i"
    _test_qual_type(statement._variable_type, TypeQualifier.TEMP, NumericalType, PrimitiveDataType.FLOAT32)

    binary = statement._expression
    assert isinstance(binary, BinaryExpression), "Expected an BinaryExpression"
    assert binary._operation == BinaryOperation.MULTIPLICATION, "Expected * BinaryOperator Operation"

    print("What is the Left Expression: ", binary._left_expression)

    # TODO: This part is Failing, since we are not Accurately Assigning Float Literals.
    assert isinstance(binary._left_expression, FloatLiteral), "Expected FloatLiteral Expression"
    assert binary._left_expression.value == 5.0, "Expected FloatLiteral Value of float(5.0)"
