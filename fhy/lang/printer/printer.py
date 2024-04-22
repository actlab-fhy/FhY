from typing import Any, Callable
from fhy.lang import ast
from fhy.lang.ast.visitor import BasePass
from fhy import ir


class ASTPrettyPrinter(BasePass):
    _is_identifier_id_printed: bool
    _indent_char: str
    _current_indent: int

    def __init__(self, indent_char: str, is_identifier_id_printed: bool) -> None:
        super().__init__()
        self._is_identifier_id_printed = is_identifier_id_printed
        self._indent_char = indent_char
        self._current_indent = 0

    def _increment_indent(self) -> None:
        self._current_indent += 1

    def _decrement_indent(self) -> None:
        if self._current_indent <= 0:
            raise ValueError("Indent cannot be negative")
        self._current_indent -= 1

    def _format_statements(self, statements: list[str]) -> str:
        return "\n".join(f"{self._indent_char * self._current_indent}{line}" for line in statements)

    def visit_Module(self, module: ast.Module) -> str:
        return "\n".join(
            self.visit(component) for component in module.components
        )

    def visit_Operation(self, operation: ast.Operation) -> str:
        self._increment_indent()
        pprinted_statements = self._format_statements([self.visit(statement) for statement in operation.body])
        self._decrement_indent()
        return f"op {self.visit(operation.name)}({', '.join(self.visit(arg) for arg in operation.args)}) -> {self.visit(operation.return_type)} " + "{\n" + pprinted_statements + "\n}"

    def visit_Procedure(self, procedure: ast.Procedure) -> str:
        self._increment_indent()
        pprinted_statements = self._format_statements([self.visit(statement) for statement in procedure.body])
        self._decrement_indent()
        return f"proc {self.visit(procedure.name)}({', '.join(self.visit(arg) for arg in procedure.args)}) " + "{\n" + pprinted_statements + "\n}"

    def visit_Argument(self, argument: ast.Argument) -> str:
        return f"{self.visit(argument.qualified_type)} {self.visit(argument.name)}"

    def visit_DeclarationStatement(self, declaration_statement: ast.DeclarationStatement) -> str:
        left = f"{self.visit(declaration_statement.variable_type)} {self.visit(declaration_statement.variable_name)}"
        if declaration_statement.expression is not None:
            right = f" = {self.visit(declaration_statement.expression)};"
        else:
            right = ";"
        return left + right

    def visit_ExpressionStatement(self, expression_statement: ast.ExpressionStatement) -> str:
        if expression_statement.left is not None:
            left = f"{self.visit(expression_statement.left)} = "
        else:
            left = ""
        return left + self.visit(expression_statement.right) + ";"

    def visit_ReturnStatement(self, return_statement: ast.ReturnStatement) -> str:
        return f"return {self.visit(return_statement.expression)};"

    def visit_UnaryExpression(self, unary_expression: ast.UnaryExpression) -> str:
        return f"{unary_expression.operation}({self.visit(unary_expression.expression)})"

    def visit_BinaryExpression(self, binary_expression: ast.BinaryExpression) -> str:
        return f"({self.visit(binary_expression.left)} {binary_expression.operation} {self.visit(binary_expression.right)})"

    def visit_TernaryExpression(self, ternary_expression: ast.TernaryExpression) -> str:
        return f"({self.visit(ternary_expression.condition)} ? {self.visit(ternary_expression.true)} : {self.visit(ternary_expression.false)})"

    def visit_FunctionExpression(self, function_expression: ast.FunctionExpression) -> str:
        template_types = ", ".join(self.visit(template_type) for template_type in function_expression.template_types)
        indices = ", ".join(self.visit(index) for index in function_expression.indices)
        args = ", ".join(self.visit(arg) for arg in function_expression.args)
        return f"{self.visit(function_expression.function)}<{template_types}>[{indices}]({args})"

    def visit_ArrayAccessExpression(self, array_access_expression: ast.ArrayAccessExpression) -> str:
        return f"{self.visit(array_access_expression.array_expression)}[{', '.join(self.visit(index) for index in array_access_expression.indices)}]"

    def visit_IdentifierExpression(self, identifier_expression: ast.IdentifierExpression) -> str:
        return self.visit(identifier_expression.identifier)

    def visit_IntLiteral(self, int_literal: ast.IntLiteral) -> str:
        return str(int_literal.value)

    def visit_QualifiedType(self, qualified_type: ast.QualifiedType) -> str:
        return f"{qualified_type.type_qualifier} {self.visit(qualified_type.base_type)}"

    def visit_NumericalType(self, numerical_type: ir.NumericalType) -> str:
        if len(numerical_type.shape) == 0:
            shape = ""
        else:
            shape = f"[{', '.join(self.visit(dim) for dim in numerical_type.shape)}]"
        return f"{self.visit(numerical_type.data_type)}{shape}"

    def visit_DataType(self, data_type: ir.DataType) -> str:
        return str(data_type.primitive_data_type)

    def visit_IndexType(self, index_type: ir.IndexType) -> str:
        index_range = f"{self.visit(index_type.lower_bound)}:{self.visit(index_type.upper_bound)}"
        if index_type.stride is not None:
            index_range += f":{self.visit(index_type.stride)}"
        else:
            index_range += ":1"
        return f"index[{index_range}]"

    def visit_Identifier(self, identifier: ir.Identifier) -> str:
        if self._is_identifier_id_printed:
            return f"({identifier.name_hint}::{identifier.id})"
        else:
            return identifier.name_hint

    def default(self, node: ast.ASTNode) -> str:
        return f"{type(node).__name__} NOT IMPLEMENTED"



def pprint_ast(ast: ast.ASTNode, indent_char: str = "  ", is_identifier_id_printed: bool = False) -> str:
    pprinter = ASTPrettyPrinter(indent_char, is_identifier_id_printed)
    return pprinter(ast)
