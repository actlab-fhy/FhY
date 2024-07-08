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

"""Pretty print serialization of AST nodes into FhY language.

Functions:
    pprint_ast: Helper function to serialize the AST node into text

Classes:
    ASTPrettyPrinter: Deconstructs AST nodes into FhY language text

"""

from fhy import ir
from fhy.lang import ast
from fhy.lang.ast.alias import ASTObject
from fhy.lang.ast.visitor import BasePass


class ASTPrettyFormatter(BasePass):
    """Deconstructs an AST node back into FhY language text using a visitor pattern.

    Args:
        indent_char (str): character(s) used to indent the output text
        show_id (bool): Include assigned identifier ID in output text if true

    Raises:
        RuntimeError: When class is improperly used and the indent becomes negative.

    """

    show_id: bool
    _indent_char: str
    _current_indent: int

    def __init__(self, indent_char: str, show_id: bool) -> None:
        super().__init__()
        self.show_id = show_id
        self._indent_char = indent_char
        self._current_indent = 0

    @property
    def _spacer(self):
        """Current indentation spacer."""
        return self._indent_char * self._current_indent

    def _increment_indent(self) -> None:
        self._current_indent += 1

    def _decrement_indent(self) -> None:
        if self._current_indent <= 0:
            raise RuntimeError("Indent cannot be negative")
        self._current_indent -= 1

    def _format_statements(self, statements: list[str]) -> str:
        return "\n".join(f"{self._spacer}{line}" for line in statements)

    def visit_Module(self, module: ast.Module) -> str:
        return "\n".join(self.visit(statement) for statement in module.statements)

    def visit_Import(self, node: ast.Import) -> str:
        return "import " + self.visit(node.name) + ";"

    def visit_Operation(self, operation: ast.Operation) -> str:
        self._increment_indent()
        pprinted_statements = self._format_statements(
            [self.visit(statement) for statement in operation.body]
        )
        self._decrement_indent()
        name = self.visit(operation.name)
        templates = ", ".join(self.visit(arg) for arg in operation.templates)
        args = ", ".join(self.visit(arg) for arg in operation.args)
        ret = self.visit(operation.return_type)

        return (
            f"op {name}<{templates}>({args}) -> {ret} "
            + "{\n"
            + pprinted_statements
            + "\n}"
        )

    def visit_Procedure(self, procedure: ast.Procedure) -> str:
        self._increment_indent()
        pprinted_statements = self._format_statements(
            [self.visit(statement) for statement in procedure.body]
        )
        self._decrement_indent()
        name = self.visit(procedure.name)
        templates = ", ".join(self.visit(arg) for arg in procedure.templates)
        args = ", ".join(self.visit(arg) for arg in procedure.args)

        return (
            f"proc {name}<{templates}>({args}) " + "{\n" + pprinted_statements + "\n}"
        )

    def visit_Argument(self, argument: ast.Argument) -> str:
        return f"{self.visit(argument.qualified_type)} {self.visit(argument.name)}"

    def visit_DeclarationStatement(
        self, declaration_statement: ast.DeclarationStatement
    ) -> str:
        left_type = self.visit(declaration_statement.variable_type)
        left_name = self.visit(declaration_statement.variable_name)
        left = f"{left_type} {left_name}"
        if declaration_statement.expression is not None:
            right = f" = {self.visit(declaration_statement.expression)};"
        else:
            right = ";"

        return left + right

    def visit_ExpressionStatement(
        self, expression_statement: ast.ExpressionStatement
    ) -> str:
        if expression_statement.left is not None:
            left = f"{self.visit(expression_statement.left)} = "
        else:
            left = ""

        return left + self.visit(expression_statement.right) + ";"

    def visit_SelectionStatement(
        self, selection_statement: ast.SelectionStatement
    ) -> str:
        self._increment_indent()
        true_body = self._format_statements(
            [self.visit(statement) for statement in selection_statement.true_body]
        )
        false_body = self._format_statements(
            [self.visit(statement) for statement in selection_statement.false_body]
        )
        self._decrement_indent()
        condition = self.visit(selection_statement.condition)

        text = f"if {condition} " + "{\n" + true_body + f"\n{self._spacer}" + "}"

        if len(false_body) > 0:
            text += " else {\n" + false_body + f"\n{self._spacer}" + "}"

        return text

    def visit_ForAllStatement(self, for_all_statement: ast.ForAllStatement) -> str:
        self._increment_indent()
        pprinted_body = self._format_statements(
            [self.visit(statement) for statement in for_all_statement.body]
        )
        self._decrement_indent()
        index = self.visit(for_all_statement.index)

        return f"forall ({index}) " + "{\n" + pprinted_body + f"\n{self._spacer}" + "}"

    def visit_ReturnStatement(self, return_statement: ast.ReturnStatement) -> str:
        return f"return {self.visit(return_statement.expression)};"

    def visit_UnaryExpression(self, unary_expression: ast.UnaryExpression) -> str:
        return (
            f"{unary_expression.operation.value}"
            f"({self.visit(unary_expression.expression)})"
        )

    def visit_BinaryExpression(self, binary_expression: ast.BinaryExpression) -> str:
        left = self.visit(binary_expression.left)
        right = self.visit(binary_expression.right)

        return f"({left} {binary_expression.operation.value} {right})"

    def visit_TernaryExpression(self, ternary_expression: ast.TernaryExpression) -> str:
        condition = self.visit(ternary_expression.condition)
        _true = self.visit(ternary_expression.true)
        _false = self.visit(ternary_expression.false)

        return f"({condition} ? {_true} : {_false})"

    def visit_FunctionExpression(
        self, function_expression: ast.FunctionExpression
    ) -> str:
        template_types = ", ".join(
            self.visit(template_type)
            for template_type in function_expression.template_types
        )
        indices = ", ".join(self.visit(index) for index in function_expression.indices)
        args = ", ".join(self.visit(arg) for arg in function_expression.args)
        func = self.visit(function_expression.function)

        return f"{func}<{template_types}>[{indices}]({args})"

    def _build_base_tuple(self, nodes: list[ASTObject]) -> str:
        a: str = "( " + ", ".join([self.visit(i) for i in nodes])
        a += ", )" if len(nodes) == 1 else " )"

        return a

    def visit_TupleExpression(self, node: ast.TupleExpression) -> str:
        return self._build_base_tuple(node.expressions)

    def visit_TupleAccessExpression(self, node: ast.TupleAccessExpression) -> str:
        _tuple: str = self.visit(node.tuple_expression)
        element: str = self.visit_IntLiteral(node.element_index)

        return f"{_tuple}.{element}"

    def visit_ArrayAccessExpression(
        self, array_access_expression: ast.ArrayAccessExpression
    ) -> str:
        index = ", ".join(
            self.visit(index) for index in array_access_expression.indices
        )
        return f"{self.visit(array_access_expression.array_expression)}[{index}]"

    def visit_IdentifierExpression(
        self, identifier_expression: ast.IdentifierExpression
    ) -> str:
        return self.visit(identifier_expression.identifier)

    def visit_IntLiteral(self, int_literal: ast.IntLiteral) -> str:
        return str(int_literal.value)

    def visit_FloatLiteral(self, float_literal: ast.FloatLiteral) -> str:
        return str(float_literal.value)

    def visit_ComplexLiteral(self, complex_literal: ast.ComplexLiteral) -> str:
        return str(complex_literal.value)

    def visit_QualifiedType(self, qualified_type: ast.QualifiedType) -> str:
        return (
            f"{qualified_type.type_qualifier.value} "
            f"{self.visit(qualified_type.base_type)}"
        )

    def visit_NumericalType(self, numerical_type: ir.NumericalType) -> str:
        if len(numerical_type.shape) == 0:
            shape = ""
        else:
            shape = f"[{', '.join(self.visit(dim) for dim in numerical_type.shape)}]"

        return f"{self.visit(numerical_type.data_type)}{shape}"

    def visit_PrimitiveDataType(self, node: ir.PrimitiveDataType) -> str:
        return str(node.primitive_data_type.value)

    def visit_TemplateDataType(self, node: ir.TemplateDataType) -> str:
        return self.visit_Identifier(node.template_type)

    def visit_IndexType(self, index_type: ir.IndexType) -> str:
        index_range = f"{self.visit(index_type.lower_bound)}:"
        index_range += f"{self.visit(index_type.upper_bound)}:"

        if index_type.stride is not None:
            index_range += f"{self.visit(index_type.stride)}"
        else:
            index_range += "1"

        return f"index[{index_range}]"

    def visit_TupleType(self, tuple_type: ir.TupleType) -> str:
        return "tuple " + self._build_base_tuple(tuple_type._types)

    def visit_Identifier(self, identifier: ir.Identifier) -> str:
        if self.show_id:
            return f"({identifier.name_hint}::{identifier.id})"
        else:
            return identifier.name_hint

    def default(self, node: ast.ASTNode) -> str:
        return f"{type(node).__name__} NOT IMPLEMENTED"


def pformat_ast(
    ast: ast.ASTNode, indent_char: str = "  ", show_id: bool = False
) -> str:
    """Returns FhY text from a given an AST node.

    Args:
        ast (ASTNode): a valid FhY AST node
        indent_char (str): character(s) used to indent the output text
        show_id (bool): Include assigned ID in output text if true

    Raises:
        RuntimeError: When class is improperly used and the indent becomes negative.

    """
    pformatter = ASTPrettyFormatter(indent_char, show_id)
    return pformatter(ast)
