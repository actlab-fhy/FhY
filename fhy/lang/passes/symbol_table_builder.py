"""Methods to construct a Symbol Table from an AST Node.

Functions:
    build_symbol_table: Primary Entry Point to construct a Symbol Table from an ASTnode.

Classes:
    SymbolTableBuilder: The Workhorse behind building a symbol table from AST

"""

from typing import Any, Set

from fhy import ir
from fhy.lang import ast
from fhy.lang.ast.visitor import Visitor
from fhy.utils import Stack
from fhy.utils.error import FhYSemanticsError

from .identifier_collector import collect_identifiers


# TODO: when visitor class automatically visits children, remove the
#       super().visit_xxx(node) call
class SymbolTableBuilder(Visitor):
    """Builds a symbol table for the given AST module node.

    The class will throw an exception if a variable is used before being declared or if
    a variable is declared more than once within the same namespace.

    Raises:
        FhYSemanticsError: A variable is used before being declared (undefined), or
            the variable is defined again (redefined), within the current namespace.
        RuntimeError: Unexpected Behavior, indicating improper use.
        TypeError: Received wrong argument (node) type.

    """

    _symbol_table: ir.SymbolTable

    _table_stack: Stack[ir.Table[ir.Identifier, ir.SymbolTableFrame]]

    def __init__(self) -> None:
        super().__init__()

        self._symbol_table = ir.SymbolTable()

        self._table_stack = Stack[ir.Table[ir.Identifier, ir.SymbolTableFrame]]()

    @property
    def symbol_table(
        self,
    ) -> ir.SymbolTable:
        return self._symbol_table

    def _push_namespace(self, namespace_name_hint: str) -> None:
        namespace_name = ir.Identifier(namespace_name_hint)
        self._symbol_table[namespace_name] = ir.Table[
            ir.Identifier, ir.SymbolTableFrame
        ]()
        self._table_stack.push(self._symbol_table[namespace_name])

    def _pop_namespace(self) -> None:
        self._table_stack.pop()

    def _assert_symbol_not_defined(self, symbol: ir.Identifier) -> None:
        if self._is_symbol_defined(symbol):
            msg = "Symbol Identifier previously declared (redefined) in current"
            raise FhYSemanticsError(f"{msg} namespace: {symbol.name_hint}")

    def _assert_symbol_defined(self, symbol: ir.Identifier) -> None:
        if not self._is_symbol_defined(symbol):
            msg = "Undeclared Symbol Identifier used in current namespace"
            raise FhYSemanticsError(f"{msg}: {symbol.name_hint}")

    def _is_symbol_defined(self, symbol: ir.Identifier) -> bool:
        for table in self._table_stack:
            if symbol in table.keys():
                return True
        return False

    def _add_symbol(self, symbol: ir.Identifier, frame: ir.SymbolTableFrame) -> None:
        if len(self._table_stack) == 0:
            raise RuntimeError(
                "Expected current table to be set before adding a symbol to it"
            )
        self._table_stack.peek()[symbol] = frame

    def __call__(self, node: ast.Module, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(node, ast.Module):
            raise TypeError(f"Expected a `Module` node. Received: {type(node)}")
        return super().__call__(node, *args, **kwargs)

    def visit_Module(self, node: ast.Module) -> None:
        self._push_namespace(node.name.name_hint)
        super().visit_Module(node)
        self._pop_namespace()

        if len(self._table_stack) != 0:
            raise RuntimeError(
                "Expected the table stack to be empty after visiting module node."
            )

    def visit_Import(self, node: ast.Import) -> None:
        self._assert_symbol_not_defined(node.name)
        import_frame = ir.ImportSymbolTableFrame(name=node.name)
        self._add_symbol(node.name, import_frame)
        super().visit_Import(node)

    def visit_Procedure(self, node: ast.Procedure) -> None:
        self._assert_symbol_not_defined(node.name)
        proc_frame = ir.FunctionSymbolTableFrame(
            name=node.name,
            signature=[
                (arg.qualified_type.type_qualifier, arg.qualified_type.base_type)
                for arg in node.args
            ],
        )
        self._add_symbol(node.name, proc_frame)
        self._push_namespace(node.name.name_hint)
        super().visit_Procedure(node)
        self._pop_namespace()

    def visit_Operation(self, node: ast.Operation) -> None:
        self._assert_symbol_not_defined(node.name)
        op_frame = ir.FunctionSymbolTableFrame(
            name=node.name,
            signature=[
                (arg.qualified_type.type_qualifier, arg.qualified_type.base_type)
                for arg in node.args
            ]
            + [(node.return_type.type_qualifier, node.return_type.base_type)],
        )
        self._add_symbol(node.name, op_frame)
        self._push_namespace(node.name.name_hint)
        super().visit_Operation(node)
        self._pop_namespace()

    def visit_Argument(self, node: ast.Argument) -> None:
        arg_frame = ir.VariableSymbolTableFrame(
            name=node.name,
            type=node.qualified_type.base_type,
            type_qualifier=node.qualified_type.type_qualifier,
        )
        self._add_symbol(node.name, arg_frame)

        if isinstance(node.qualified_type.base_type, ir.NumericalType):
            shape_dimension_identifiers: Set[ir.Identifier] = set()
            for shape in node.qualified_type.base_type.shape:
                shape_dimension_identifiers.update(collect_identifiers(shape))

            for dimension in shape_dimension_identifiers:
                if not self._is_symbol_defined(dimension):
                    var_frame = ir.VariableSymbolTableFrame(
                        name=dimension,
                        type=ir.NumericalType(ir.DataType(ir.PrimitiveDataType.INT)),
                        type_qualifier=ir.TypeQualifier.PARAM,
                    )
                    self._add_symbol(dimension, var_frame)

        super().visit_Argument(node)

    def visit_DeclarationStatement(self, node: ast.DeclarationStatement) -> None:
        self._assert_symbol_not_defined(node.variable_name)
        var_frame = ir.VariableSymbolTableFrame(
            name=node.variable_name,
            type=node.variable_type.base_type,
            type_qualifier=node.variable_type.type_qualifier,
        )
        self._add_symbol(node.variable_name, var_frame)
        super().visit_DeclarationStatement(node)

    def visit_IdentifierExpression(self, node: ast.IdentifierExpression) -> None:
        self._assert_symbol_defined(node.identifier)
        super().visit_IdentifierExpression(node)


def build_symbol_table(node: ast.Module) -> ir.SymbolTable:
    """Build a symbol table from a module AST node.

    Argument:
        node (ast.Module): FhY Module AST Node

    Returns:
        (ir.SymbolTable) Symbol table cataloging all variables from the provided module,
            by appropriate frame.

    Raises:
        UndeclaredIdentifierException: A variable is used before being Declared.
        AlreadyDeclaredIdentifierException: A variable is declared more than once within
            the same namespace.

        FhYSemanticsError: A variable is used before being declared (undefined), or
            the variable is defined again (redefined), within the current namespace.
        RuntimeError: Unexpected Behavior, indicating improper use.
        TypeError: Received wrong argument (node) type.

    """
    builder = SymbolTableBuilder()
    builder(node)

    return builder.symbol_table
