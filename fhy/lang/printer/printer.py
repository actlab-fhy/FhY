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

    def __call__(self, node: ast.ASTNode) -> Any:
        return self.pprint(node)

    def _increment_indent(self) -> None:
        self._current_indent += 1

    def _decrement_indent(self) -> None:
        if self._current_indent <= 0:
            raise ValueError("Indent cannot be negative")
        self._current_indent -= 1

    def _format_statements(self, statements: list[str]) -> str:
        return "\n".join(f"{self._indent_char * self._current_indent}{line}" for line in statements)

    def pprint(self, node: ast.ASTNode) -> str:
        method: Callable[[ast.ASTNode], str] = self.default
        for cls in type(node).mro():
            if issubclass(cls, ast.ASTNode):
                name: str = "pprint_" + cls.keyname()
                if hasattr(self, name):
                    method = getattr(self, name)
        return method(node)

    def default(self, node: ast.ASTNode) -> str:
        return f"{node.keyname()} NOT IMPLEMENTED"

    def pprint_Module(self, module: ast.Module) -> str:
        return "\n".join(
            self.pprint(component) for component in module.components
        )

    def pprint_Operation(self, operation: ast.Operation) -> str:
        self._increment_indent()
        pprinted_statements = self._format_statements([self.pprint(statement) for statement in operation.body])
        self._decrement_indent()
        return f"op {self.pprint_Identifier(operation.name)}({', '.join(self.pprint(arg) for arg in operation.args)}) -> {self.pprint(operation.return_type)} " + "{\n" + pprinted_statements + "\n}"

    def pprint_Procedure(self, procedure: ast.Procedure) -> str:
        self._increment_indent()
        pprinted_statements = self._format_statements([self.pprint(statement) for statement in procedure.body])
        self._decrement_indent()
        return f"proc {self.pprint_Identifier(procedure.name)}({', '.join(self.pprint(arg) for arg in procedure.args)}) " + "{\n" + pprinted_statements + "\n}"

    def pprint_Argument(self, argument: ast.Argument) -> str:
        return f"{self.pprint(argument.qualified_type)} {self.pprint_Identifier(argument.name)}"

    def pprint_DeclarationStatement(self, declaration_statement: ast.DeclarationStatement) -> str:
        left = f"{self.pprint(declaration_statement.variable_type)} {self.pprint_Identifier(declaration_statement.variable_name)}"
        if declaration_statement.expression is not None:
            right = f" = {self.pprint(declaration_statement.expression)};"
        else:
            right = ";"
        return left + right

    def pprint_ExpressionStatement(self, expression_statement: ast.ExpressionStatement) -> str:
        if expression_statement.left is not None:
            left = f"{self.pprint(expression_statement.left)} = "
        else:
            left = ""
        return left + self.pprint(expression_statement.right) + ";"

    def pprint_ReturnStatement(self, return_statement: ast.ReturnStatement) -> str:
        return f"return {self.pprint(return_statement.expression)};"

    def pprint_UnaryExpression(self, unary_expression: ast.UnaryExpression) -> str:
        return f"{unary_expression.operation}({self.pprint(unary_expression.expression)})"

    def pprint_BinaryExpression(self, binary_expression: ast.BinaryExpression) -> str:
        return f"({self.pprint(binary_expression.left)} {binary_expression.operation} {self.pprint(binary_expression.right)})"

    def pprint_TernaryExpression(self, ternary_expression: ast.TernaryExpression) -> str:
        return f"({self.pprint(ternary_expression.condition)} ? {self.pprint(ternary_expression.true)} : {self.pprint(ternary_expression.false)})"

    def pprint_FunctionExpression(self, function_expression: ast.FunctionExpression) -> str:
        template_types = ", ".join(self.pprint_Type(template_type) for template_type in function_expression.template_types)
        indices = ", ".join(self.pprint(index) for index in function_expression.indices)
        args = ", ".join(self.pprint(arg) for arg in function_expression.args)
        return f"{self.pprint(function_expression.function)}<{template_types}>[{indices}]({args})"

    def pprint_ArrayAccessExpression(self, array_access_expression: ast.ArrayAccessExpression) -> str:
        return f"{self.pprint(array_access_expression.array_expression)}[{', '.join(self.pprint(index) for index in array_access_expression.indices)}]"

    def pprint_IdentifierExpression(self, identifier_expression: ast.IdentifierExpression) -> str:
        return self.pprint_Identifier(identifier_expression.identifier)

    def pprint_IntLiteral(self, int_literal: ast.IntLiteral) -> str:
        return str(int_literal.value)

    def pprint_QualifiedType(self, qualified_type: ast.QualifiedType) -> str:
        return f"{qualified_type.type_qualifier} {self.pprint_Type(qualified_type.base_type)}"

    def pprint_Type(self, type: ir.Type) -> str:
        if isinstance(type, ir.NumericalType):
            return self.pprint_NumericalType(type)
        elif isinstance(type, ir.IndexType):
            return self.pprint_IndexType(type)
        else:
            raise Exception()

    def pprint_NumericalType(self, numerical_type: ir.NumericalType) -> str:
        if len(numerical_type.shape) == 0:
            shape = ""
        else:
            shape = f"[{', '.join(self.pprint(dim) for dim in numerical_type.shape)}]"
        return f"{self.pprint_DataType(numerical_type.data_type)}{shape}"

    def pprint_DataType(self, data_type: ir.DataType) -> str:
        return self.pprint_PrimitiveDataType(data_type.primitive_data_type)

    def pprint_PrimitiveDataType(self, primitive_data_type: ir.PrimitiveDataType) -> str:
        return str(primitive_data_type)

    def pprint_IndexType(self, index_type: ir.IndexType) -> str:
        index_range = f"{self.pprint(index_type.lower_bound)}:{self.pprint(index_type.upper_bound)}"
        if index_type.stride is not None:
            index_range += f":{self.pprint(index_type.stride)}"
        else:
            index_range += ":1"
        return f"index[{index_range}]"

    def pprint_Identifier(self, identifier: ir.Identifier) -> str:
        if self._is_identifier_id_printed:
            return f"({identifier.name_hint}::{identifier._id})"
        else:
            return identifier.name_hint


def pprint_ast(ast: ast.ASTNode, indent_char: str = "  ", is_identifier_id_printed: bool = False) -> str:
    pprinter = ASTPrettyPrinter(indent_char, is_identifier_id_printed)
    return pprinter(ast)
