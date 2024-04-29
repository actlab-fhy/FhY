"""This is a simplistic unit test, testing behavior of ASTNodes."""

from typing import List, Type

import pytest

from fhy.lang import ast


@pytest.mark.parametrize(
    "node, expected",
    [
        (ast.ASTNode, "ASTNode"),
        # Core
        (ast.Module, "Module"),
        (ast.Component, "Component"),
        (ast.Function, "Function"),
        (ast.Statement, "Statement"),
        (ast.Expression, "Expression"),
        # Components
        (ast.Import, "Import"),
        (ast.Argument, "Argument"),
        (ast.Procedure, "Procedure"),
        (ast.Operation, "Operation"),
        (ast.Native, "Native"),
        # Expression
        (ast.UnaryExpression, "UnaryExpression"),
        (ast.BinaryExpression, "BinaryExpression"),
        (ast.TernaryExpression, "TernaryExpression"),
        (ast.TupleAccessExpression, "TupleAccessExpression"),
        (ast.FunctionExpression, "FunctionExpression"),
        (ast.ArrayAccessExpression, "ArrayAccessExpression"),
        (ast.TupleExpression, "TupleExpression"),
        (ast.IdentifierExpression, "IdentifierExpression"),
        (ast.Literal, "Literal"),
        (ast.IntLiteral, "IntLiteral"),
        (ast.FloatLiteral, "FloatLiteral"),
        (ast.ComplexLiteral, "ComplexLiteral"),
        # Statement
        (ast.DeclarationStatement, "DeclarationStatement"),
        (ast.ExpressionStatement, "ExpressionStatement"),
        (ast.ForAllStatement, "ForAllStatement"),
        (ast.SelectionStatement, "SelectionStatement"),
        (ast.ReturnStatement, "ReturnStatement"),
        # Qualified Type
        (ast.QualifiedType, "QualifiedType"),
    ],
)
def test_keynames(node: Type[ast.ASTNode], expected: str):
    """Tests Expected Key name of an ASTNode."""
    result = node.keyname()
    assert result == expected, f"Unexpected Key name of ASTNode: {result} | {expected}"


@pytest.mark.parametrize(
    "node, is_abstract, expected",
    [
        (ast.ASTNode, True, ["span"]),
        # Core
        (ast.Module, False, ["span", "components"]),
        (ast.Component, True, ["span"]),
        (ast.Function, True, ["span", "name"]),
        (ast.Statement, True, ["span"]),
        (ast.Expression, True, ["span"]),
        # Components
        (ast.Import, False, ["span", "name"]),
        (ast.Argument, False, ["span", "name", "qualified_type"]),
        (ast.Procedure, False, ["span", "name", "args", "body"]),
        (ast.Operation, False, ["span", "name", "args", "body", "return_type"]),
        (ast.Native, False, ["span", "name", "args"]),
        # Expression
        (ast.UnaryExpression, False, ["span", "operation", "expression"]),
        (ast.BinaryExpression, False, ["span", "operation", "left", "right"]),
        (ast.TernaryExpression, False, ["span", "condition", "true", "false"]),
        (
            ast.TupleAccessExpression,
            False,
            ["span", "tuple_expression", "element_index"],
        ),
        (
            ast.FunctionExpression,
            False,
            ["span", "function", "template_types", "indices", "args"],
        ),
        (ast.ArrayAccessExpression, False, ["span", "array_expression", "indices"]),
        (ast.TupleExpression, False, ["span", "expressions"]),
        (ast.IdentifierExpression, False, ["span", "identifier"]),
        (ast.Literal, True, ["span"]),
        (ast.IntLiteral, False, ["span", "value"]),
        (ast.FloatLiteral, False, ["span", "value"]),
        (ast.ComplexLiteral, False, ["span", "value"]),
        # Statement
        (
            ast.DeclarationStatement,
            False,
            ["span", "variable_name", "variable_type", "expression"],
        ),
        (ast.ExpressionStatement, False, ["span", "left", "right"]),
        (ast.ForAllStatement, False, ["span", "index", "body"]),
        (
            ast.SelectionStatement,
            False,
            ["span", "condition", "true_body", "false_body"],
        ),
        (ast.ReturnStatement, False, ["span", "expression"]),
        # Qualified Type
        (ast.QualifiedType, False, ["span", "base_type", "type_qualifier"]),
    ],
)
def test_attributes(node: type[ast.ASTNode], is_abstract: bool, expected: List[str]):
    """Test that the attributes trickle down subclasses."""
    # Mock Instantiate
    if not is_abstract:
        instance = node(**{key: None for key in expected})

    else:

        class Test(node):
            def visit_attrs(self) -> List[str]:
                return super().visit_attrs()

        instance = Test(**{key: None for key in expected})

    assert set(instance.visit_attrs()) == set(
        expected
    ), f"Unexpected Attributes: {node}"


def test_node_equality():
    """Testing Special Dunder Method, __eq__, behavior."""

    class Derived(ast.ASTNode):
        def visit_attrs(self) -> List[str]:
            return super().visit_attrs()

    a = Derived()
    b = Derived()

    assert a == b, "Expected AST Nodes to be equivalent."
