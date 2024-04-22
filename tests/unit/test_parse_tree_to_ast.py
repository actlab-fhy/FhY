import pytest
from typing import Any, Callable, List, Optional, Type
from antlr4 import (
    BailErrorStrategy,
    CommonTokenStream,
    InputStream,
    ParserRuleContext,
    RecognitionException,
)
from antlr4.error.ErrorListener import ErrorListener

from fhy import ir
from fhy.lang import ast
from fhy.lang.ast_builder import from_parse_tree
from fhy.lang.parser import FhYLexer, FhYParser


# TODO: make all identifier name equality not in terms of name hint after scope and loading identifiers with table is implemented


# TODO: move to a utils module; only for error messages from assertions
def list_to_types(xs: List[Any]) -> List[type]:
    return [type(x) for x in xs]


def is_primitive_expression_equal(expr1: ast.Expression, expr2: ast.Expression) -> bool:
    primitive_expression_types = (ast.IntLiteral, ast.FloatLiteral,
                                  ast.IdentifierExpression, ast.TupleExpression,
                                  ast.TupleAccessExpression,
                                  ast.ArrayAccessExpression)
    if not isinstance(expr1, primitive_expression_types) or not isinstance(expr2, primitive_expression_types):
        raise ValueError(f"Both expressions must be primitive expressions: {type(expr1)}, {type(expr2)}")

    if isinstance(expr1, ast.IntLiteral) and isinstance(expr2, ast.IntLiteral):
        return expr1.value == expr2.value
    elif isinstance(expr1, ast.FloatLiteral) and isinstance(expr2, ast.FloatLiteral):
        return expr1.value == expr2.value
    elif isinstance(expr1, ast.IdentifierExpression) and isinstance(expr2, ast.IdentifierExpression):
        # TODO: remove the name hint portion once a more robust table for pulling identifiers in the same scope is created
        return expr1.identifier.name_hint == expr2.identifier.name_hint
    elif isinstance(expr1, ast.TupleExpression) and isinstance(expr2, ast.TupleExpression):
        raise NotImplementedError()
    elif isinstance(expr1, ast.TupleAccessExpression) and isinstance(expr2, ast.TupleAccessExpression):
        raise NotImplementedError()
    elif isinstance(expr1, ast.ArrayAccessExpression) and isinstance(expr2, ast.ArrayAccessExpression):
        if len(expr1.indices) != len(expr2.indices):
            return False
        for expr1_index, expr2_index in zip(expr1.indices, expr2.indices):
            if not is_primitive_expression_equal(expr1_index, expr2_index):
                return False
        return True
    else:
        return False


class ThrowingErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise SyntaxError(f"Syntax error at {line}:{column} - {msg}")


@pytest.fixture(scope="module")
def lexer() -> Callable[[str], FhYLexer]:
    def create_lexer(input_str: str):
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


def _parse_file_contents(parser, file_contents: str) -> ParserRuleContext:
    parse_tree = parser(file_contents).module()
    assert parse_tree is not None, "Expected parse tree for module"
    return parse_tree


def _assert_is_expected_module(
    node: ast.ASTNode,
    expected_num_components: int
) -> None:
    assert isinstance(node, ast.Module), f"Expected \"Module\" AST node, got \"{type(node)}\""
    assert all(isinstance(component, ast.Component) for component in node.components), f"Expected all components to be \"Component\" AST nodes, got \"{list_to_types(node.components)}\""
    assert len(node.components) == expected_num_components, f"Expected module to have {expected_num_components} components"


def _assert_is_expected_procedure(
    node: ast.ASTNode,
    expected_name: str,
    expected_num_args: int,
    expected_num_statements: int
) -> None:
    assert isinstance(node, ast.Procedure), f"Expected \"Procedure\" AST node, got \"{type(node)}\""
    assert isinstance(node.name, ir.Identifier), f"Expected procedure name to be \"Identifier\", got \"{type(node.name)}\""
    assert node.name.name_hint == expected_name, f"Expected procedure name to be \"{expected_name}\", got \"{node.name.name_hint}\""
    assert all(isinstance(arg, ast.Argument) for arg in node.args), f"Expected all arguments to be \"Argument\" AST nodes, got \"{list_to_types(node.args)}\""
    assert len(node.args) == expected_num_args, f"Expected procedure to have {expected_num_args} arguments, got {len(node.args)}"
    assert all(isinstance(statement, ast.Statement) for statement in node.body), f"Expected all statements to be \"Statement\" AST nodes, got \"{list_to_types(node.body)}\""
    assert len(node.body) == expected_num_statements, f"Expected procedure to have {expected_num_statements} statements, got {len(node.body)}"


def _assert_is_expected_operation(
    node: ast.ASTNode,
    expected_name: str,
    expected_num_args: int,
    expected_num_statements: int
) -> None:
    assert isinstance(node, ast.Operation), f"Expected \"Operation\" AST node, got \"{type(node)}\""
    assert isinstance(node.name, ir.Identifier), f"Expected operation name to be \"Identifier\", got \"{type(node.name)}\""
    assert node.name.name_hint == expected_name, f"Expected operation name to be \"{expected_name}\", got \"{node.name.name_hint}\""
    assert all(isinstance(arg, ast.Argument) for arg in node.args), f"Expected all arguments to be \"Argument\" AST nodes, got \"{list_to_types(node.args)}\""
    assert len(node.args) == expected_num_args, f"Expected operation to have {expected_num_args} arguments, got {len(node.args)}"
    assert all(isinstance(statement, ast.Statement) for statement in node.body), f"Expected all statements to be \"Statement\" AST nodes, got \"{list_to_types(node.body)}\""
    assert len(node.body) == expected_num_statements, f"Expected operation to have {expected_num_statements} statements, got {len(node.body)}"


def _assert_is_expected_qualified_type(
    node: ast.ASTNode,
    expected_type_qualifier: ir.TypeQualifier,
    expected_base_type_cls: Type[ir.Type]
) -> None:
    assert isinstance(node, ast.QualifiedType), f"Expected \"QualifiedType\" AST node, got \"{type(node)}\""
    assert node.type_qualifier == expected_type_qualifier, f"Expected type qualifier to be \"{expected_type_qualifier}\", got \"{node.type_qualifier}\""
    assert isinstance(node.base_type, expected_base_type_cls), f"Expected base type to be \"{expected_base_type_cls}\", got \"{type(node.base_type)}\""


def _assert_is_expected_argument(
    node: ast.ASTNode,
    expected_name: str,
) -> None:
    assert isinstance(node, ast.Argument), f"Expected \"Argument\" AST node, got \"{type(node)}\""
    assert isinstance(node.name, ir.Identifier), f"Expected argument name to be \"Identifier\", got \"{type(node.name)}\""
    assert node.name.name_hint == expected_name, f"Expected argument name to be \"{expected_name}\", got \"{node.name.name_hint}\""


def _assert_is_expected_numerical_type(
    numerical_type: ir.NumericalType,
    expected_primitive_data_type: ir.PrimitiveDataType,
    expected_shape: List[ast.Expression]
) -> None:
    assert isinstance(numerical_type, ir.NumericalType), f"Expected \"NumericalType\", got \"{type(numerical_type)}\""
    assert numerical_type.data_type.primitive_data_type == expected_primitive_data_type, f"Expected primitive data type to be \"{expected_primitive_data_type}\", got \"{numerical_type.data_type.primitive_data_type}\""
    assert all(isinstance(expr, ast.Expression) for expr in numerical_type.shape), f"Expected all shape components to be \"Expression\" AST nodes, got \"{list_to_types(numerical_type.shape)}\""
    assert len(numerical_type.shape) == len(expected_shape), f"Expected numerical type shape to have {len(expected_shape)} components, got {len(numerical_type.shape)}"
    for i, shape_component in enumerate(numerical_type.shape):
        assert is_primitive_expression_equal(shape_component, expected_shape[i]), f"Expected shape component {i} to be equal (expected: {expected_shape[i]}, actual: {shape_component})"


def _assert_is_expected_shape(
    shape: List[ast.Expression],
    expected_shape: List[ast.Expression]
) -> None:
    assert isinstance(shape, list), f"Expected shape to be a list, got \"{type(shape)}\""
    assert all(isinstance(expr, ast.Expression) for expr in shape), f"Expected all shape components to be \"Expression\" AST nodes, got \"{list_to_types(shape)}\""
    assert len(shape) == len(expected_shape), f"Expected shape to have {len(expected_shape)} components, got {len(shape)}"
    for i, shape_component in enumerate(shape):
        assert is_primitive_expression_equal(shape_component, expected_shape[i]), f"Expected shape component {i} to be equal (expected: {expected_shape[i]}, actual: {shape_component})"


def _assert_is_expected_index_type(
    index_type: ir.IndexType,
    expected_low: ast.Expression,
    expected_high: ast.Expression,
    expected_stride: Optional[ast.Expression]
) -> None:
    assert isinstance(index_type, ir.IndexType), f"Expected \"IndexType\", got \"{type(index_type)}\""
    assert isinstance(index_type.lower_bound, ast.Expression), f"Expected lower bound to be \"Expression\" AST node, got \"{type(index_type.lower_bound)}\""
    assert is_primitive_expression_equal(index_type.lower_bound, expected_low), f"Expected lower bound to be equal (expected: {expected_low}, actual: {index_type.lower_bound})"
    assert isinstance(index_type.upper_bound, ast.Expression), f"Expected upper bound to be \"Expression\" AST node, got \"{type(index_type.upper_bound)}\""
    assert is_primitive_expression_equal(index_type.upper_bound, expected_high), f"Expected upper bound to be equal (expected: {expected_high}, actual: {index_type.upper_bound})"
    if expected_stride is not None:
        assert isinstance(index_type.stride, ast.Expression), f"Expected stride to be \"Expression\" AST node, got \"{type(index_type.stride)}\""
        assert is_primitive_expression_equal(index_type.stride, expected_stride), f"Expected stride to be equal (expected: {expected_stride}, actual: {index_type.stride})"


def _assert_is_expected_declaration_statement(
    node: ast.ASTNode,
    expected_variable_name: ir.Identifier,
    expected_expression: Optional[ast.Expression]
) -> None:
    assert isinstance(node, ast.DeclarationStatement), f"Expected \"DeclarationStatement\" AST node, got \"{type(node)}\""
    assert isinstance(node.variable_name, ir.Identifier), f"Expected variable name to be \"Identifier\", got \"{type(node.variable_name)}\""
    assert node.variable_name.name_hint == expected_variable_name.name_hint, f"Expected variable name to be \"{expected_variable_name.name_hint}\", got \"{node.variable_name.name_hint}\""
    assert isinstance(node.variable_type, ast.QualifiedType), f"Expected variable type to be \"QualifiedType\", got \"{type(node.variable_type)}\""
    if node.expression is not None:
        assert isinstance(node.expression, ast.Expression), f"Expected expression to be \"Expression\" AST node, got \"{type(node.expression)}\""
    if expected_expression is not None:
        assert is_primitive_expression_equal(node.expression, expected_expression), f"Expected expression to be \"{expected_expression}\", got \"{node.expression}\""


def _assert_is_expected_expression_statement(
    node: ast.ASTNode,
    expected_left_expression: Optional[ast.Expression],
    expected_right_expression: ast.Expression
) -> None:
    assert isinstance(node, ast.ExpressionStatement), f"Expected \"ExpressionStatement\" AST node, got \"{type(node)}\""
    if expected_left_expression is not None:
        assert isinstance(node.left, ast.Expression), f"Expected left expression to be \"Expression\" AST node, got \"{type(node.left)}\""
        assert is_primitive_expression_equal(node.left, expected_left_expression), f"Expected left expression to be equal (expected: {expected_left_expression}, actual: {node.left})"
    assert isinstance(node.right, ast.Expression), f"Expected right expression to be \"Expression\" AST node, got \"{type(node.right)}\""
    assert is_primitive_expression_equal(node.right, expected_right_expression), f"Expected right expression to be equal (expected: {expected_right_expression}, actual: {node.right})"


def _assert_is_expected_return_statement(
    node: ast.ASTNode,
    expected_expression: ast.Expression
) -> None:
    assert isinstance(node, ast.ReturnStatement), f"Expected \"ReturnStatement\" AST node, got \"{type(node)}\""
    assert isinstance(node.expression, ast.Expression), f"Expected expression to be \"Expression\" AST node, got \"{type(node.expression)}\""
    assert is_primitive_expression_equal(node.expression, expected_expression), f"Expected expression to be equal (expected: {expected_expression}, actual: {node.expression})"


def test_empty_file(parser):
    """Test that an empty file is converted correctly."""
    source_file_content = ""
    parse_tree = _parse_file_contents(parser, source_file_content)

    _ast = from_parse_tree(parse_tree)

    assert isinstance(_ast, ast.Module), "Expected \"Module\" AST node"
    assert len(_ast.components) == 0, "Expected empty module"


def test_empty_procedure(parser):
    """Test that an empty procedure is converted correctly."""
    source_file_content = "proc foo(){}"
    parse_tree = _parse_file_contents(parser, source_file_content)

    _ast = from_parse_tree(parse_tree)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "foo", 0, 0)


def test_empty_procedure_with_qualified_argument(parser):
    """Test that an empty procedure with a single qualified
    argument is converted correctly.

    """
    source_file_content = "proc foo(input int32 x){}"
    parse_tree = _parse_file_contents(parser, source_file_content)

    _ast = from_parse_tree(parse_tree)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "foo", 1, 0)

    arg = procedure.args[0]
    _assert_is_expected_argument(arg, "x")

    arg_qualified_type = arg.qualified_type
    _assert_is_expected_qualified_type(arg_qualified_type, ir.TypeQualifier.INPUT, ir.NumericalType)
    arg_base_type = arg_qualified_type.base_type
    _assert_is_expected_numerical_type(arg_base_type, ir.PrimitiveDataType.INT32, [])


def test_empty_procedure_with_a_qualified_argument_with_shape(parser):
    source_file_content = "proc foo(input int32[m, n] x){}"
    parse_tree = _parse_file_contents(parser, source_file_content)

    _ast = from_parse_tree(parse_tree)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "foo", 1, 0)

    arg = procedure.args[0]
    _assert_is_expected_argument(arg, "x")

    arg_qualified_type = arg.qualified_type
    _assert_is_expected_qualified_type(arg_qualified_type, ir.TypeQualifier.INPUT, ir.NumericalType)
    arg_type_shape = arg_qualified_type.base_type.shape
    _assert_is_expected_shape(arg_type_shape, [ast.IdentifierExpression(span=None, identifier=ir.Identifier("m")), ast.IdentifierExpression(span=None, identifier=ir.Identifier("n"))])


def test_empty_operation(parser):
    """test that an Empty Operation is Converted Correctly"""
    source_file_content = "op foo(){}"
    parse_tree = parser(source_file_content).module()
    assert parse_tree is not None

    _ast = from_parse_tree(parse_tree)

    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "foo", 0, 0)


def test_empty_operation_return_type(parser):
    """Tests that an Empty Operation with a Return Type is Converted Correctly"""
    source_file_content = "op foo(input int32[n, m] x) -> output int32[n, m] {}"
    parse_tree = parser(source_file_content).module()

    _ast = from_parse_tree(parse_tree)

    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "foo", 1, 0)

    arg = operation.args[0]
    _assert_is_expected_argument(arg, "x")

    arg_qualified_type = arg.qualified_type
    _assert_is_expected_qualified_type(arg_qualified_type, ir.TypeQualifier.INPUT, ir.NumericalType)
    arg_base_type: ir.Type = arg_qualified_type.base_type
    _assert_is_expected_numerical_type(arg_base_type, ir.PrimitiveDataType.INT32, [ast.IdentifierExpression(span=None, identifier=ir.Identifier("n")), ast.IdentifierExpression(span=None, identifier=ir.Identifier("m"))])

    return_type = operation.return_type
    _assert_is_expected_qualified_type(return_type, ir.TypeQualifier.OUTPUT, ir.NumericalType)
    return_type_shape = return_type.base_type.shape
    _assert_is_expected_shape(return_type_shape, [ast.IdentifierExpression(span=None, identifier=ir.Identifier("n")), ast.IdentifierExpression(span=None, identifier=ir.Identifier("m"))])


def test_declaration_statement(parser):
    """Tests a single Delcaration Statement."""
    source_file_content = "proc foo(){temp int32 i;}"
    parse_tree = parser(source_file_content).module()

    _ast = from_parse_tree(parse_tree)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "foo", 0, 1)

    statement = procedure.body[0]
    _assert_is_expected_declaration_statement(statement, ir.Identifier("i"), None)

    statement_qualified_type = statement.variable_type
    _assert_is_expected_qualified_type(statement_qualified_type, ir.TypeQualifier.TEMP, ir.NumericalType)
    statement_qualified_type_shape = statement_qualified_type.base_type.shape
    _assert_is_expected_shape(statement_qualified_type_shape, [])


def test_return_statement(parser):
    """Tests a Return Statement"""
    source_file_content = "op foo() -> temp int32 {temp int32 i = 5; return i;}"
    parse_tree = _parse_file_contents(parser, source_file_content)

    _ast = from_parse_tree(parse_tree)

    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "foo", 0, 2)

    statement = operation.body[1]
    _assert_is_expected_return_statement(statement, ast.IdentifierExpression(span=None, identifier=ir.Identifier("i")))


def test_unary_expressions(parser):
    """Tests a Unary Expression (Negative)"""
    source_file_content = "op foo() -> temp int32 {temp int32 i = -5;}"
    parse_tree = _parse_file_contents(parser, source_file_content)

    _ast = from_parse_tree(parse_tree)

    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "foo", 0, 1)

    statement = operation.body[0]
    assert isinstance(statement, ast.DeclarationStatement), f"Expected \"DeclarationStatement\" AST node, got \"{type(statement)}\""

    statement_expression = statement.expression
    assert isinstance(statement_expression, ast.UnaryExpression), f"Expected \"UnaryExpression\" AST node, got \"{type(statement_expression)}\""
    assert statement_expression.operation == ast.UnaryOperation.NEGATIVE, f"Expected operation to be \"{ast.UnaryOperation.NEGATIVE}\", got \"{statement_expression.operation}\""
    statement_expression_operand = statement_expression.expression
    assert isinstance(statement_expression_operand, ast.IntLiteral), f"Expected operand to be \"IntLiteral\" AST node, got \"{type(statement_expression_operand)}\""
    assert statement_expression_operand.value == 5, f"Expected operand value to be 5, got {statement_expression_operand.value}"


def test_binary_expressions(parser):
    """Tests a Binary Expression (Multiplication)"""
    source_file_content = "op foo() -> temp float32 {temp float32 i = 5 * 6;}"
    parse_tree = _parse_file_contents(parser, source_file_content)

    _ast = from_parse_tree(parse_tree)

    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "foo", 0, 1)

    statement = operation.body[0]
    assert isinstance(statement, ast.DeclarationStatement), f"Expected \"DeclarationStatement\" AST node, got \"{type(statement)}\""

    statement_expression = statement.expression
    assert isinstance(statement_expression, ast.BinaryExpression), f"Expected \"BinaryExpression\" AST node, got \"{type(statement_expression)}\""
    assert statement_expression.operation == ast.BinaryOperation.MULTIPLICATION, f"Expected operation to be \"{ast.BinaryOperation.MULTIPLICATION}\", got \"{statement_expression.operation}\""
    statement_expression_left = statement_expression.left
    assert isinstance(statement_expression_left, ast.IntLiteral), f"Expected left expression to be \"IntLiteral\" AST node, got \"{type(statement_expression_left)}\""
    assert statement_expression_left.value == 5, f"Expected left expression value to be 5, got {statement_expression_left.value}"
    statement_expression_right = statement_expression.right
    assert isinstance(statement_expression_right, ast.IntLiteral), f"Expected right expression to be \"IntLiteral\" AST node, got \"{type(statement_expression_right)}\""
    assert statement_expression_right.value == 6, f"Expected right expression value to be 6, got {statement_expression_right.value}"


def test_ternary_expressions(parser):
    """Tests a Ternary Conditional Expression"""
    source_file_content = "op foo() {temp float32 i = 5 < 6 ? 7 : 8;}"
    parse_tree = _parse_file_contents(parser, source_file_content)

    _ast = from_parse_tree(parse_tree)

    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "foo", 0, 1)

    statement = operation.body[0]
    assert isinstance(statement, ast.DeclarationStatement), f"Expected \"DeclarationStatement\" AST node, got \"{type(statement)}\""

    statement_expression = statement.expression
    assert isinstance(statement_expression, ast.TernaryExpression), f"Expected \"TernaryExpression\" AST node, got \"{type(statement_expression)}\""
    statement_expression_condition = statement_expression.condition
    assert isinstance(statement_expression_condition, ast.BinaryExpression), f"Expected condition to be \"BinaryExpression\" AST node, got \"{type(statement_expression_condition)}\""
    assert statement_expression_condition.operation == ast.BinaryOperation.LESS_THAN, f"Expected condition operation to be \"{ast.BinaryOperation.LESS_THAN}\", got \"{statement_expression_condition.operation}\""
    statement_expression_condition_left = statement_expression_condition.left
    assert isinstance(statement_expression_condition_left, ast.IntLiteral), f"Expected condition left expression to be \"IntLiteral\" AST node, got \"{type(statement_expression_condition_left)}\""
    assert statement_expression_condition_left.value == 5, f"Expected condition left expression value to be 5, got {statement_expression_condition_left.value}"
    statement_expression_condition_right = statement_expression_condition.right
    assert isinstance(statement_expression_condition_right, ast.IntLiteral), f"Expected condition right expression to be \"IntLiteral\" AST node, got \"{type(statement_expression_condition_right)}\""
    assert statement_expression_condition_right.value == 6, f"Expected condition right expression value to be 6, got {statement_expression_condition_right.value}"
    statement_expression_true = statement_expression.true
    assert isinstance(statement_expression_true, ast.IntLiteral), f"Expected true expression to be \"IntLiteral\" AST node, got \"{type(statement_expression_true)}\""
    assert statement_expression_true.value == 7, f"Expected true expression value to be 7, got {statement_expression_true.value}"
    statement_expression_false = statement_expression.false
    assert isinstance(statement_expression_false, ast.IntLiteral), f"Expected false expression to be \"IntLiteral\" AST node, got \"{type(statement_expression_false)}\""
    assert statement_expression_false.value == 8, f"Expected false expression value to be 8, got {statement_expression_false.value}"


def test_index_type(parser):
    """Test Construction of an Index Type"""
    source_file_content = "proc bar() {temp index[1:m] i;}"
    parse_tree = _parse_file_contents(parser, source_file_content)

    _ast = from_parse_tree(parse_tree)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "bar", 0, 1)

    statement = procedure.body[0]
    assert isinstance(statement, ast.DeclarationStatement), f"Expected \"DeclarationStatement\" AST node, got \"{type(statement)}\""

    statement_qualified_type = statement.variable_type
    _assert_is_expected_qualified_type(statement_qualified_type, ir.TypeQualifier.TEMP, ir.IndexType)
    statement_qualified_type_base_type = statement_qualified_type.base_type
    _assert_is_expected_index_type(statement_qualified_type_base_type, ast.IntLiteral(span=None, value=1), ast.IdentifierExpression(span=None, identifier=ir.Identifier("m")), None)


def test_function_expression(parser):
    source_file_content = "proc bar() {temp int32 i = foo(A);}"
    parse_tree = _parse_file_contents(parser, source_file_content)

    _ast = from_parse_tree(parse_tree)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "bar", 0, 1)

    statement = procedure.body[0]
    assert isinstance(statement, ast.DeclarationStatement), f"Expected \"DeclarationStatement\" AST node, got \"{type(statement)}\""

    statement_expression = statement.expression
    assert isinstance(statement_expression, ast.FunctionExpression), f"Expected \"FunctionExpression\" AST node, got \"{type(statement_expression)}\""
    statement_expression_function = statement_expression.function
    assert isinstance(statement_expression_function, ast.IdentifierExpression), f"Expected function to be \"IdentifierExpression\" AST node, got \"{type(statement_expression_function)}\""
    assert statement_expression_function.identifier.name_hint == "foo", f"Expected function name to be \"foo\", got \"{statement_expression_function.identifier.name_hint}\""
    assert len(statement_expression.template_types) == 0, f"Expected no template types, got {len(statement_expression.template_types)}"
    assert len(statement.expression.indices) == 0, f"Expected no indices, got {len(statement.expression.indices)}"
    assert len(statement_expression.args) == 1, f"Expected 1 argument, got {len(statement_expression.args)}"
    statement_expression_arg = statement_expression.args[0]
    assert isinstance(statement_expression_arg, ast.IdentifierExpression), f"Expected argument to be \"IdentifierExpression\" AST node, got \"{type(statement_expression_arg)}\""
    assert statement_expression_arg.identifier.name_hint == "A", f"Expected argument name to be \"A\", got \"{statement_expression_arg.identifier.name_hint}\""


def test_tensor_access_expressions(parser):
    """Tests construction of TensorAccess Expressions."""
    source_file_content = "proc bar() {A[i] = 1;}"
    parse_tree = _parse_file_contents(parser, source_file_content)

    _ast = from_parse_tree(parse_tree)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "bar", 0, 1)

    statement = procedure.body[0]
    _assert_is_expected_expression_statement(statement, ast.ArrayAccessExpression(span=None, array_expression=ast.IdentifierExpression(span=None, identifier=ir.Identifier("A")), indices=[ast.IdentifierExpression(span=None, identifier=ir.Identifier("i"))]), ast.IntLiteral(span=None, value=1))

