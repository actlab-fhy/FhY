"""Test Conversion of FhY Source Code from CST to AST."""

from typing import List, Optional, Type

import pytest

from fhy import ir
from fhy.lang import ast
from fhy.utils import error

from ..utils import list_to_types

# TODO: make all identifier name equality not in terms of name hint after scope and
#       loading identifiers with table is implemented


def wrong_node_babe(node_a, node_b) -> str:
    """Wrong Node Babe.

    What you see, is what you get. And what you got, is unexpected...

    """
    name = node_a.__name__
    return f"Expected `{name}` AST node, got `{type(node_b)}`"


def is_primitive_expression_equal(expr1: ast.Expression, expr2: ast.Expression) -> bool:
    """Confirm Equality Between Two Primitive Expression Types."""
    primitive_expression_types = (
        ast.IntLiteral,
        ast.FloatLiteral,
        ast.IdentifierExpression,
        ast.TupleExpression,
        ast.TupleAccessExpression,
        ast.ArrayAccessExpression,
    )
    if not isinstance(expr1, primitive_expression_types) or not isinstance(
        expr2, primitive_expression_types
    ):
        raise ValueError(
            "Both expressions must be primitive expressions: "
            f"{type(expr1)}, {type(expr2)}"
        )

    if isinstance(expr1, ast.IntLiteral) and isinstance(expr2, ast.IntLiteral):
        return expr1.value == expr2.value
    elif isinstance(expr1, ast.FloatLiteral) and isinstance(expr2, ast.FloatLiteral):
        return expr1.value == expr2.value
    elif isinstance(expr1, ast.IdentifierExpression) and isinstance(
        expr2, ast.IdentifierExpression
    ):
        # TODO: remove the name hint portion once a more robust table for pulling
        #       identifiers in the same scope is created
        return expr1.identifier.name_hint == expr2.identifier.name_hint
    elif isinstance(expr1, ast.TupleExpression) and isinstance(
        expr2, ast.TupleExpression
    ):
        raise NotImplementedError()
    elif isinstance(expr1, ast.TupleAccessExpression) and isinstance(
        expr2, ast.TupleAccessExpression
    ):
        raise NotImplementedError()
    elif isinstance(expr1, ast.ArrayAccessExpression) and isinstance(
        expr2, ast.ArrayAccessExpression
    ):
        if len(expr1.indices) != len(expr2.indices):
            return False
        for expr1_index, expr2_index in zip(expr1.indices, expr2.indices):
            if not is_primitive_expression_equal(expr1_index, expr2_index):
                return False
        return True
    else:
        return False


def _assert_is_expected_module(node: ast.ASTNode, expected_num_components: int) -> None:
    assert isinstance(node, ast.Module), wrong_node_babe(ast.Module, node)

    assert all(isinstance(component, ast.Component) for component in node.components), (
        "Expected all components to be `Component` AST nodes, got "
        + f"`{list_to_types(node.components)}`"
    )
    assert (
        len(node.components) == expected_num_components
    ), f"Expected module to have {expected_num_components} components"


def _assert_is_expected_import(node: ast.ASTNode, expected_import: str) -> None:
    assert isinstance(node, ast.Import), wrong_node_babe(ast.Import, node)

    assert isinstance(
        node.name, ir.Identifier
    ), f'Expected import name to be "Identifier", got "{type(node.name)}"'
    assert (
        node.name.name_hint == expected_import
    ), f'Expected import name to be "{expected_import}", got "{node.name.name_hint}"'


def _assert_is_expected_procedure(
    node: ast.ASTNode,
    expected_name: str,
    expected_num_args: int,
    expected_num_statements: int,
) -> None:
    assert isinstance(node, ast.Procedure), wrong_node_babe(ast.Procedure, node)

    assert isinstance(
        node.name, ir.Identifier
    ), f'Expected procedure name to be "Identifier", got "{type(node.name)}"'
    assert (
        node.name.name_hint == expected_name
    ), f'Expected procedure name to be "{expected_name}", got "{node.name.name_hint}"'
    assert all(isinstance(arg, ast.Argument) for arg in node.args), (
        "Expected all arguments to be `Argument` AST nodes, got "
        + f"`{list_to_types(node.args)}`"
    )
    assert len(node.args) == expected_num_args, (
        f"Expected procedure to have {expected_num_args} arguments, got "
        + f"{len(node.args)}"
    )
    assert all(isinstance(statement, ast.Statement) for statement in node.body), (
        "Expected all statements to be `Statement` AST nodes, got "
        + f"{list_to_types(node.body)}"
    )
    assert len(node.body) == expected_num_statements, (
        f"Expected procedure to have {expected_num_statements} statements, got "
        + f"{len(node.body)}"
    )


def _assert_is_expected_operation(
    node: ast.ASTNode,
    expected_name: str,
    expected_num_args: int,
    expected_num_statements: int,
) -> None:
    assert isinstance(node, ast.Operation), wrong_node_babe(ast.Operation, node)

    assert isinstance(
        node.name, ir.Identifier
    ), f'Expected operation name to be "Identifier", got "{type(node.name)}"'
    assert (
        node.name.name_hint == expected_name
    ), f'Expected operation name to be "{expected_name}", got "{node.name.name_hint}"'
    assert all(isinstance(arg, ast.Argument) for arg in node.args), (
        "Expected all arguments to be `Argument` AST nodes, got "
        + f"`{list_to_types(node.args)}`"
    )
    assert len(node.args) == expected_num_args, (
        f"Expected operation to have {expected_num_args} arguments, got "
        + f"{len(node.args)}"
    )
    assert all(isinstance(statement, ast.Statement) for statement in node.body), (
        "Expected all statements to be `Statement` AST nodes, got "
        + f"`{list_to_types(node.body)}`"
    )
    assert len(node.body) == expected_num_statements, (
        f"Expected operation to have {expected_num_statements} statements, got "
        + f"{len(node.body)}"
    )


def _assert_is_expected_qualified_type(
    node: ast.ASTNode,
    expected_type_qualifier: ir.TypeQualifier,
    expected_base_type_cls: Type[ir.Type],
) -> None:
    assert isinstance(node, ast.QualifiedType), wrong_node_babe(ast.QualifiedType, node)

    assert node.type_qualifier == expected_type_qualifier, (
        f"Expected type qualifier to be `{expected_type_qualifier}`, "
        + f"got `{node.type_qualifier}`"
    )
    assert isinstance(node.base_type, expected_base_type_cls), (
        f'Expected base type to be "{expected_base_type_cls}", '
        + f"got `{type(node.base_type)}`"
    )


def _assert_is_expected_argument(
    node: ast.ASTNode,
    expected_name: str,
) -> None:
    assert isinstance(node, ast.Argument), wrong_node_babe(ast.Argument, node)

    assert isinstance(node.name, ir.Identifier), wrong_node_babe(
        ir.Identifier, node.name
    )
    assert (
        node.name.name_hint == expected_name
    ), f'Expected argument name to be "{expected_name}", got "{node.name.name_hint}"'


def _assert_is_expected_numerical_type(
    numerical_type: ir.NumericalType,
    expected_primitive_data_type: ir.PrimitiveDataType,
    expected_shape: List[ast.Expression],
) -> None:
    assert isinstance(numerical_type, ir.NumericalType), wrong_node_babe(
        ir.NumericalType, numerical_type
    )

    assert (
        numerical_type.data_type.primitive_data_type == expected_primitive_data_type
    ), (
        f"Expected primitive data type to be `{expected_primitive_data_type}`, got "
        + f"`{numerical_type.data_type.primitive_data_type}`"
    )

    assert all(isinstance(expr, ast.Expression) for expr in numerical_type.shape), (
        "Expected all shape components to be `Expression` AST nodes, got "
        + f"`{list_to_types(numerical_type.shape)}`"
    )
    assert len(numerical_type.shape) == len(expected_shape), (
        f"Expected numerical type shape to have {len(expected_shape)} components, got "
        + f"{len(numerical_type.shape)}"
    )
    for i, shape_component in enumerate(numerical_type.shape):
        assert is_primitive_expression_equal(shape_component, expected_shape[i]), (
            f"Expected shape component {i} to be equal "
            + f"(expected: {expected_shape[i]}, actual: {shape_component})"
        )


def _assert_is_expected_shape(
    shape: List[ast.Expression], expected_shape: List[ast.Expression]
) -> None:
    assert isinstance(shape, list), f'Expected shape to be a list, got "{type(shape)}"'
    assert all(isinstance(expr, ast.Expression) for expr in shape), (
        "Expected all shape components to be `Expression` AST nodes, got "
        + f"`{list_to_types(shape)}`"
    )
    assert len(shape) == len(
        expected_shape
    ), f"Expected shape to have {len(expected_shape)} components, got {len(shape)}"
    for i, shape_component in enumerate(shape):
        assert is_primitive_expression_equal(shape_component, expected_shape[i]), (
            f"Expected shape component {i} to be equal "
            + f"(expected: {expected_shape[i]}, actual: {shape_component})"
        )


def _assert_is_expected_index_type(
    index_type: ir.IndexType,
    expected_low: ast.Expression,
    expected_high: ast.Expression,
    expected_stride: Optional[ast.Expression],
) -> None:
    assert isinstance(index_type, ir.IndexType), wrong_node_babe(
        ir.IndexType, index_type
    )

    assert isinstance(index_type.lower_bound, ast.Expression), wrong_node_babe(
        ast.Expression, index_type.lower_bound
    )

    assert is_primitive_expression_equal(index_type.lower_bound, expected_low), (
        "Expected lower bound to be equal "
        + f"(expected: {expected_low}, actual: {index_type.lower_bound})"
    )
    assert isinstance(index_type.upper_bound, ast.Expression), wrong_node_babe(
        ast.Expression, index_type.upper_bound
    )

    assert is_primitive_expression_equal(index_type.upper_bound, expected_high), (
        "Expected upper bound to be equal "
        + f"(expected: {expected_high}, actual: {index_type.upper_bound})"
    )
    if expected_stride is not None:
        assert isinstance(index_type.stride, ast.Expression), wrong_node_babe(
            ast.Expression, index_type.stride
        )
        assert is_primitive_expression_equal(index_type.stride, expected_stride), (
            "Expected stride to be equal "
            + f"(expected: {expected_stride}, actual: {index_type.stride})"
        )


def _assert_is_expected_declaration_statement(
    node: ast.ASTNode,
    expected_variable_name: ir.Identifier,
    expected_expression: Optional[ast.Expression],
) -> None:
    assert isinstance(node, ast.DeclarationStatement), wrong_node_babe(
        ast.DeclarationStatement, node
    )

    assert isinstance(node.variable_name, ir.Identifier), wrong_node_babe(
        ir.Identifier, node.variable_name
    )

    assert node.variable_name.name_hint == expected_variable_name.name_hint, (
        f"Expected variable name to be `{expected_variable_name.name_hint}`, "
        + f"got `{node.variable_name.name_hint}`"
    )
    assert isinstance(node.variable_type, ast.QualifiedType), wrong_node_babe(
        ast.QualifiedType, node.variable_type
    )
    if node.expression is not None:
        assert isinstance(node.expression, ast.Expression), wrong_node_babe(
            ast.Expression, node.expression
        )

    if expected_expression is not None:
        assert is_primitive_expression_equal(
            node.expression, expected_expression
        ), f'Expected expression to be "{expected_expression}", got "{node.expression}"'


def _assert_is_expected_expression_statement(
    node: ast.ASTNode,
    expected_left_expression: Optional[ast.Expression],
    expected_right_expression: ast.Expression,
) -> None:
    assert isinstance(node, ast.ExpressionStatement), wrong_node_babe(
        ast.ExpressionStatement, node
    )

    if expected_left_expression is not None:
        assert isinstance(node.left, ast.Expression), wrong_node_babe(
            ast.Expression, node.left
        )

        assert is_primitive_expression_equal(node.left, expected_left_expression), (
            "Expected left expression to be equal "
            + f"(expected: {expected_left_expression}, actual: {node.left})"
        )
    assert isinstance(node.right, ast.Expression), wrong_node_babe(
        ast.Expression, node.right
    )
    assert is_primitive_expression_equal(node.right, expected_right_expression), (
        "Expected right expression to be equal "
        + f"(expected: {expected_right_expression}, actual: {node.right})"
    )


def _assert_is_expected_return_statement(
    node: ast.ASTNode, expected_expression: ast.Expression
) -> None:
    assert isinstance(node, ast.ReturnStatement), wrong_node_babe(
        ast.ReturnStatement, node
    )

    assert isinstance(node.expression, ast.Expression), wrong_node_babe(
        ast.Expression, node.expression
    )
    assert is_primitive_expression_equal(node.expression, expected_expression), (
        "Expected expression to be equal "
        + f"(expected: {expected_expression}, actual: {node.expression})"
    )


def test_empty_file(construct_ast):
    """Test that an empty file is converted correctly."""
    source_file_content = ""
    _ast = construct_ast(source_file_content)

    assert isinstance(_ast, ast.Module), wrong_node_babe(ast.Module, _ast)
    assert len(_ast.components) == 0, "Expected empty module"


def test_empty_procedure(construct_ast):
    """Test that an empty procedure is converted correctly."""
    source_file_content = "proc foo(){}"
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "foo", 0, 0)


def test_empty_procedure_with_qualified_argument(construct_ast):
    """Test that an empty procedure with a single qualified argument."""
    source_file_content = "proc foo(input int32 x){}"
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "foo", 1, 0)

    arg = procedure.args[0]
    _assert_is_expected_argument(arg, "x")

    arg_qualified_type = arg.qualified_type
    _assert_is_expected_qualified_type(
        arg_qualified_type, ir.TypeQualifier.INPUT, ir.NumericalType
    )
    arg_base_type = arg_qualified_type.base_type
    _assert_is_expected_numerical_type(arg_base_type, ir.PrimitiveDataType.INT32, [])


def test_empty_procedure_with_a_qualified_argument_with_shape(construct_ast):
    """Test an Empty procedure containing Arguments with Shape."""
    source_file_content = "proc foo(input int32[m, n] x){}"
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "foo", 1, 0)

    arg = procedure.args[0]
    _assert_is_expected_argument(arg, "x")

    arg_qualified_type = arg.qualified_type
    _assert_is_expected_qualified_type(
        arg_qualified_type, ir.TypeQualifier.INPUT, ir.NumericalType
    )
    arg_type_shape = arg_qualified_type.base_type.shape
    _assert_is_expected_shape(
        arg_type_shape,
        [
            ast.IdentifierExpression(span=None, identifier=ir.Identifier("m")),
            ast.IdentifierExpression(span=None, identifier=ir.Identifier("n")),
        ],
    )


def test_empty_operation(construct_ast):
    """Test that an Empty Operation is Converted Correctly."""
    source_file_content = "op foo() -> output int32 {}"
    _ast = construct_ast(source_file_content)
    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "foo", 0, 0)


def test_empty_operation_return_type(construct_ast):
    """Test that an Empty Operation with a Return Type is Converted Correctly."""
    source_file_content = "op foo(input int32[n, m] x) -> output int32[n, m] {}"
    _ast = construct_ast(source_file_content)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "foo", 1, 0)

    arg = operation.args[0]
    _assert_is_expected_argument(arg, "x")

    arg_qualified_type = arg.qualified_type
    _assert_is_expected_qualified_type(
        arg_qualified_type, ir.TypeQualifier.INPUT, ir.NumericalType
    )
    arg_base_type: ir.Type = arg_qualified_type.base_type
    _assert_is_expected_numerical_type(
        arg_base_type,
        ir.PrimitiveDataType.INT32,
        [
            ast.IdentifierExpression(span=None, identifier=ir.Identifier("n")),
            ast.IdentifierExpression(span=None, identifier=ir.Identifier("m")),
        ],
    )

    return_type = operation.return_type
    _assert_is_expected_qualified_type(
        return_type, ir.TypeQualifier.OUTPUT, ir.NumericalType
    )
    return_type_shape = return_type.base_type.shape
    _assert_is_expected_shape(
        return_type_shape,
        [
            ast.IdentifierExpression(span=None, identifier=ir.Identifier("n")),
            ast.IdentifierExpression(span=None, identifier=ir.Identifier("m")),
        ],
    )


def test_declaration_statement(construct_ast):
    """Tests a single Delcaration Statement."""
    source_file_content = "proc foo(){temp int32 i;}"
    _ast = construct_ast(source_file_content)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "foo", 0, 1)

    statement = procedure.body[0]
    _assert_is_expected_declaration_statement(statement, ir.Identifier("i"), None)

    statement_qualified_type = statement.variable_type
    _assert_is_expected_qualified_type(
        statement_qualified_type, ir.TypeQualifier.TEMP, ir.NumericalType
    )
    statement_qualified_type_shape = statement_qualified_type.base_type.shape
    _assert_is_expected_shape(statement_qualified_type_shape, [])


def test_selection_statement(construct_ast):
    """Test an If (selection) Statement."""
    source_file_content = "proc foo() {if (1) {i = 1;} else {j = 1;}}"
    _ast = construct_ast(source_file_content)
    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "foo", 0, 1)

    statement = procedure.body[0]
    assert isinstance(statement, ast.SelectionStatement), wrong_node_babe(
        ast.SelectionStatement, statement
    )
    assert isinstance(statement.condition, ast.IntLiteral), wrong_node_babe(
        ast.IntLiteral, statement.condition
    )

    assert (
        statement.condition.value == 1
    ), f"Expected condition value to be 1, got {statement.condition.value}"

    assert (
        len(statement.true_body) == 1
    ), f"Expected true branch to have 1 statement, got {len(statement.true_body)}"
    true_branch = statement.true_body[0]
    assert isinstance(true_branch, ast.ExpressionStatement), wrong_node_babe(
        ast.ExpressionStatement, true_branch
    )

    assert isinstance(true_branch.left, ast.IdentifierExpression), wrong_node_babe(
        ast.IdentifierExpression, true_branch.left
    )

    assert true_branch.left.identifier.name_hint == "i", (
        "Expected left expression name hint to be `i`, got "
        + f"`{true_branch.left.identifier.name_hint}`"
    )
    assert isinstance(true_branch.right, ast.IntLiteral), wrong_node_babe(
        ast.IntLiteral, true_branch.right
    )

    assert (
        true_branch.right.value == 1
    ), f"Expected right expression value to be 1, got {true_branch.right.value}"

    assert (
        len(statement.false_body) == 1
    ), f"Expected false branch to have 1 statement, got {len(statement.false_body)}"
    false_branch = statement.false_body[0]
    assert isinstance(false_branch, ast.ExpressionStatement), wrong_node_babe(
        ast.ExpressionStatement, false_branch
    )
    assert isinstance(false_branch.left, ast.IdentifierExpression), wrong_node_babe(
        ast.IdentifierExpression, false_branch.left
    )
    assert false_branch.left.identifier.name_hint == "j", (
        "Expected left expression name hint to be `j`, got "
        + f"`{false_branch.left.identifier.name_hint}`"
    )
    assert isinstance(false_branch.right, ast.IntLiteral), wrong_node_babe(
        ast.IntLiteral, false_branch.right
    )
    assert (
        false_branch.right.value == 1
    ), f"Expected right expression value to be 1, got {false_branch.right.value}"


def test_for_all_statement(construct_ast):
    """Test an Iteration (For All) Statement (loop)."""
    source_file_content = "proc foo() {forall (i) {j = 1;}}"
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "foo", 0, 1)

    statement = procedure.body[0]
    assert isinstance(statement, ast.ForAllStatement), wrong_node_babe(
        ast.ForAllStatement, statement
    )

    assert isinstance(statement.index, ast.IdentifierExpression), wrong_node_babe(
        ast.IdentifierExpression, statement.index
    )

    assert statement.index.identifier.name_hint == "i", (
        'Expected index name hint to be "i", got '
        + f"`{statement.index.identifier.name_hint}`"
    )

    assert (
        len(statement.body) == 1
    ), f"Expected body to have 1 statement, got {len(statement.body)}"
    body_statement = statement.body[0]
    assert isinstance(body_statement, ast.ExpressionStatement), wrong_node_babe(
        ast.ExpressionStatement, body_statement
    )

    assert isinstance(body_statement.left, ast.IdentifierExpression), wrong_node_babe(
        ast.IdentifierExpression, body_statement.left
    )

    assert body_statement.left.identifier.name_hint == "j", (
        'Expected left expression name hint to be "j", got '
        + f"`{body_statement.left.identifier.name_hint}`"
    )
    assert isinstance(body_statement.right, ast.IntLiteral), wrong_node_babe(
        ast.IntLiteral, body_statement.right
    )

    assert (
        body_statement.right.value == 1
    ), f"Expected right expression value to be 1, got {body_statement.right.value}"


def test_return_statement(construct_ast):
    """Test a Return Statement."""
    source_file_content = "op foo() -> temp int32 {temp int32 i = 5; return i;}"
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "foo", 0, 2)

    statement = operation.body[1]
    _assert_is_expected_return_statement(
        statement, ast.IdentifierExpression(span=None, identifier=ir.Identifier("i"))
    )


def test_unary_expressions(construct_ast):
    """Tests a Unary Expression (Negative)."""
    source_file_content = "op foo() -> temp int32 {temp int32 i = -5;}"
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "foo", 0, 1)

    statement = operation.body[0]
    assert isinstance(statement, ast.DeclarationStatement), wrong_node_babe(
        ast.DeclarationStatement, statement
    )

    statement_expression = statement.expression
    assert isinstance(statement_expression, ast.UnaryExpression), wrong_node_babe(
        ast.UnaryExpression, statement_expression
    )

    assert statement_expression.operation == ast.UnaryOperation.NEGATIVE, (
        f'Expected operation to be "{ast.UnaryOperation.NEGATIVE}", got '
        + f"`{statement_expression.operation}`"
    )
    statement_expression_operand = statement_expression.expression
    assert isinstance(statement_expression_operand, ast.IntLiteral), wrong_node_babe(
        ast.IntLiteral, statement_expression_operand
    )

    assert (
        statement_expression_operand.value == 5
    ), f"Expected operand value to be 5, got {statement_expression_operand.value}"


def test_binary_expressions(construct_ast):
    """Tests a Binary Expression (Multiplication)."""
    source_file_content = "op foo() -> temp float32 {temp float32 i = 5 * 6;}"
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "foo", 0, 1)

    statement = operation.body[0]
    assert isinstance(statement, ast.DeclarationStatement), wrong_node_babe(
        ast.DeclarationStatement, statement
    )

    statement_expression = statement.expression
    assert isinstance(statement_expression, ast.BinaryExpression), wrong_node_babe(
        ast.BinaryExpression, statement_expression
    )

    assert statement_expression.operation == ast.BinaryOperation.MULTIPLICATION, (
        f'Expected operation to be "{ast.BinaryOperation.MULTIPLICATION}", got '
        + f"`{statement_expression.operation}`"
    )
    statement_expression_left = statement_expression.left
    assert isinstance(statement_expression_left, ast.IntLiteral), wrong_node_babe(
        ast.IntLiteral, statement_expression_left
    )

    assert (
        statement_expression_left.value == 5
    ), f"Expected left expression value to be 5, got {statement_expression_left.value}"
    statement_expression_right = statement_expression.right
    assert isinstance(statement_expression_right, ast.IntLiteral), wrong_node_babe(
        ast.IntLiteral, statement_expression_right
    )

    assert statement_expression_right.value == 6, (
        "Expected right expression value to be 6, got "
        + f"{statement_expression_right.value}"
    )


def test_ternary_expressions(construct_ast):
    """Tests a Ternary Conditional Expression."""
    source_file_content = "op foo() -> output int32 {temp float32 i = 5 < 6 ? 7 : 8;}"
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "foo", 0, 1)

    statement = operation.body[0]
    assert isinstance(statement, ast.DeclarationStatement), wrong_node_babe(
        ast.DeclarationStatement, statement
    )

    statement_expression = statement.expression
    assert isinstance(statement_expression, ast.TernaryExpression), wrong_node_babe(
        ast.TernaryExpression, statement_expression
    )

    statement_expression_condition = statement_expression.condition
    assert isinstance(
        statement_expression_condition, ast.BinaryExpression
    ), wrong_node_babe(ast.BinaryExpression, statement_expression_condition)

    assert statement_expression_condition.operation == ast.BinaryOperation.LESS_THAN, (
        f'Expected condition operation to be "{ast.BinaryOperation.LESS_THAN}", '
        + f"got `{statement_expression_condition.operation}`"
    )

    statement_expression_condition_left = statement_expression_condition.left
    assert isinstance(
        statement_expression_condition_left, ast.IntLiteral
    ), wrong_node_babe(ast.IntLiteral, statement_expression_condition_left)

    assert statement_expression_condition_left.value == 5, (
        "Expected condition left expression value to be 5, got "
        + f"{statement_expression_condition_left.value}"
    )
    statement_expression_condition_right = statement_expression_condition.right
    assert isinstance(
        statement_expression_condition_right, ast.IntLiteral
    ), wrong_node_babe(ast.IntLiteral, statement_expression_condition_right)

    assert statement_expression_condition_right.value == 6, (
        "Expected condition right expression value to be 6, got "
        + f"{statement_expression_condition_right.value}"
    )
    statement_expression_true = statement_expression.true
    assert isinstance(statement_expression_true, ast.IntLiteral), wrong_node_babe(
        ast.IntLiteral, statement_expression_true
    )

    assert (
        statement_expression_true.value == 7
    ), f"Expected true expression value to be 7, got {statement_expression_true.value}"
    statement_expression_false = statement_expression.false
    assert isinstance(statement_expression_false, ast.IntLiteral), wrong_node_babe(
        ast.IntLiteral, statement_expression_false
    )

    assert statement_expression_false.value == 8, (
        "Expected false expression value to be 8, got "
        + f"{statement_expression_false.value}"
    )


def test_index_type(construct_ast):
    """Test Construction of an Index Type."""
    source_file_content = "proc bar() {temp index[1:m] i;}"
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "bar", 0, 1)

    statement = procedure.body[0]
    assert isinstance(
        statement, ast.DeclarationStatement
    ), f'Expected "DeclarationStatement" AST node, got "{type(statement)}"'

    statement_qualified_type = statement.variable_type
    _assert_is_expected_qualified_type(
        statement_qualified_type, ir.TypeQualifier.TEMP, ir.IndexType
    )
    statement_qualified_type_base_type = statement_qualified_type.base_type
    _assert_is_expected_index_type(
        statement_qualified_type_base_type,
        ast.IntLiteral(span=None, value=1),
        ast.IdentifierExpression(span=None, identifier=ir.Identifier("m")),
        None,
    )


def test_function_expression(construct_ast):
    """Test Function Call Expression."""
    source_file_content = "proc bar() {temp int32 i = foo(A);}"
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "bar", 0, 1)

    statement = procedure.body[0]
    assert isinstance(statement, ast.DeclarationStatement), wrong_node_babe(
        ast.DeclarationStatement, statement
    )

    statement_expression = statement.expression
    assert isinstance(statement_expression, ast.FunctionExpression), wrong_node_babe(
        ast.FunctionExpression, statement_expression
    )

    statement_expression_function = statement_expression.function
    assert isinstance(
        statement_expression_function, ast.IdentifierExpression
    ), wrong_node_babe(ast.IdentifierExpression, statement_expression_function)

    assert statement_expression_function.identifier.name_hint == "foo", (
        'Expected function name to be "foo", got '
        + f"`{statement_expression_function.identifier.name_hint}`"
    )
    assert (
        len(statement_expression.template_types) == 0
    ), f"Expected no template types, got {len(statement_expression.template_types)}"
    assert (
        len(statement.expression.indices) == 0
    ), f"Expected no indices, got {len(statement.expression.indices)}"
    assert (
        len(statement_expression.args) == 1
    ), f"Expected 1 argument, got {len(statement_expression.args)}"
    statement_expression_arg = statement_expression.args[0]
    assert isinstance(
        statement_expression_arg, ast.IdentifierExpression
    ), wrong_node_babe(ast.IdentifierExpression, statement_expression_arg)

    assert statement_expression_arg.identifier.name_hint == "A", (
        "Expected argument name to be `A`, got "
        + f"`{statement_expression_arg.identifier.name_hint}`"
    )


def test_tensor_access_expressions(construct_ast):
    """Tests construction of TensorAccess Expressions."""
    source_file_content = "proc bar() {A[i] = 1;}"
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    procedure = _ast.components[0]
    _assert_is_expected_procedure(procedure, "bar", 0, 1)

    statement = procedure.body[0]
    _assert_is_expected_expression_statement(
        statement,
        ast.ArrayAccessExpression(
            span=None,
            array_expression=ast.IdentifierExpression(
                span=None, identifier=ir.Identifier("A")
            ),
            indices=[
                ast.IdentifierExpression(span=None, identifier=ir.Identifier("i"))
            ],
        ),
        ast.IntLiteral(span=None, value=1),
    )


def test_tuple_type(construct_ast):
    """Test Tuple Type Declaration Statement."""
    source_file_content = (
        "op bar() -> output int32[m,n] {output (int32[m, n], int32) i;}"
    )
    _ast = construct_ast(source_file_content)
    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "bar", 0, 1)

    statement = operation.body[0]
    assert isinstance(statement, ast.DeclarationStatement), wrong_node_babe(
        ast.DeclarationStatement, statement
    )
    assert statement.variable_name.name_hint == "i", "Expected Variable Name `i`"

    assert isinstance(statement.variable_type, ast.QualifiedType), wrong_node_babe(
        ast.QualifiedType, statement.variable_type
    )

    _tuple = statement.variable_type.base_type
    assert isinstance(_tuple, ir.TupleType), wrong_node_babe(ir.TupleType, _tuple)

    assert len(_tuple._types) == 2, "Expected 2 Types in TupleType Definition."
    t1, t2 = _tuple._types
    assert isinstance(t1, ir.NumericalType), wrong_node_babe(ir.NumericalType, t1)
    assert isinstance(t2, ir.NumericalType), wrong_node_babe(ir.NumericalType, t2)


def test_int_literal(construct_ast):
    """Test IntLiteral Construction."""
    source_file_content = (
        "op bar() -> output int32 {1; 0b0101; 0B01; 0x1; 0XFF; 0o1; 0O7;}"
    )
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "bar", 0, 7)

    for i, value in enumerate([1, 5, 1, 1, 255, 1, 7]):
        statement = operation.body[i]
        assert isinstance(statement, ast.ExpressionStatement), wrong_node_babe(
            ast.ExpressionStatement, statement
        )
        assert isinstance(statement.right, ast.IntLiteral), wrong_node_babe(
            ast.IntLiteral, statement.right
        )
        assert (
            statement.right.value == value
        ), f"Expected IntLiteral Value to be {value}"


def test_float_literal(construct_ast):
    """Test use of different Formats of Float Literal."""
    source_file_content = "op bar() -> output float32 {1.0; .2; 1.; 1e2; 1.2e3;}"
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    operation = _ast.components[0]
    _assert_is_expected_operation(operation, "bar", 0, 5)

    for i, value in enumerate([1.0, 0.2, 1.0, 100.0, 1200.0]):
        statement = operation.body[i]
        assert isinstance(statement, ast.ExpressionStatement), wrong_node_babe(
            ast.ExpressionStatement, statement
        )
        assert isinstance(statement.right, ast.FloatLiteral), wrong_node_babe(
            ast.FloatLiteral, statement.right
        )
        assert (
            statement.right.value == value
        ), f"Expected FloatLiteral Value to be {value}"


def test_absolute_import(construct_ast):
    """Test Absolute Import Component."""
    source_file_content = "import foo.bar;"
    _ast = construct_ast(source_file_content)

    _assert_is_expected_module(_ast, 1)

    import_component = _ast.components[0]
    _assert_is_expected_import(import_component, "foo.bar")


def test_line_comment(construct_ast):
    """Test that comments are skipped, creating an empty Module."""
    source = "//lorem ipsum dolor sit amet;"
    print("Source:", source)
    _ast = construct_ast(source)
    assert isinstance(_ast, ast.Module), "Expected to construct a Module Node."
    assert len(_ast.components) == 0, "Expected Module to be empty."


def test_procedure_with_line_comment(construct_ast):
    """Test procedure is found and constructed with line comments in the mix."""
    source = "//lorem ipsum dolor sit amet\nproc foo(input int32[m,n] A) {}"
    print("Source:", source)
    _ast = construct_ast(source)
    assert isinstance(_ast, ast.Module), "Expected to construct a Module Node."
    assert len(_ast.components) == 1, "Expected Module to contain 1 component."
    proc = _ast.components[0]
    _assert_is_expected_procedure(proc, "foo", 1, 0)

    # Procedure should be on second line
    assert (
        proc.span.line.start == 2
    ), f"Expected Procedure to be on Second Line: {proc.span.line.start}"


def test_syntax_error_no_argument_name(construct_ast):
    """Raise FhYSyntaxError when an function Argument is defined without a Name."""
    source_file_content = "op foo(input int32[m,n]) -> output int32 {}"
    with pytest.raises(error.FhYSyntaxError) as info:
        _ast = construct_ast(source_file_content)
    print(info.value)


def test_syntax_error_no_operation_name(construct_ast):
    """Raise Syntax Error when an Operation is defined without a Name."""
    source = "op (input int32[m,n] A) -> output int32 {}"
    # NOTE: This raises the Antlr Syntax Error, not from our visitor class.
    with pytest.raises(SyntaxError) as info:
        _ast = construct_ast(source)
    print(info.value)


def test_syntax_error_no_operation_return_type(construct_ast):
    """Raise FhYSyntaxError when an Operation is defined without a return type."""
    source = "op func(input int32[m,n] A) {}"
    with pytest.raises(error.FhYSyntaxError) as info:
        _ast = construct_ast(source)
    print(info.value)


# TODO: Corner Case - Have this raise an Error.
@pytest.mark.skip(reason="Bug. Expected Syntax Error, but Creates Empty Module")
def test_invalid_function_keyword(construct_ast):
    """This interesting bit creates an Empty Module... Instead of Raising an Error.

    This occurs because we hard code Function Keywords within the Grammar. In other
    words, without these keywords, we can never parse a function. Because it doesn't
    remotely match the syntax of anything else, it is simply bypassed by Antlr.

    """
    source = "def foo(input int32[m,n] A) -> output int32[m,n] {}"
    with pytest.raises(error.FhYSyntaxError) as info:
        _ast = construct_ast(source)
    print(_ast.__class__)
    print(_ast.components)


@pytest.mark.skip(reason="Bug. Expected Syntax Error, but Creates Empty Module")
def test_gibberish(construct_ast):
    """Nonsense gibberish (that is not a comment) should raise Errors."""
    source = "lorem ipsum dolor sit amet;"
    with pytest.raises(error.FhYSyntaxError) as info:
        _ast = construct_ast(source)
    print(_ast.__class__)
    print(_ast.components)
