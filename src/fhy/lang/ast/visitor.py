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

"""Visitor patterns to visit AST nodes.

Classes:
    BasePass: Abstract visitor pattern base class.
    Visitor: Visitor with control for how to visit child nodes.
    Listener: Visitor without control for how to visit child nodes.
    Transformer: Visitor to modify AST nodes.

"""

from abc import ABC
from copy import copy
from typing import Any, Callable, List, Sequence, Union

from fhy.ir.identifier import Identifier as IRIdentifier
from fhy.ir.type import DataType as IRDataType
from fhy.ir.type import IndexType as IRIndexType
from fhy.ir.type import NumericalType as IRNumericalType
from fhy.ir.type import PrimitiveDataType as IRPrimitiveDataType
from fhy.ir.type import Type as IRType
from fhy.ir.type import TypeQualifier as IRTypeQualifier
from fhy.lang.ast.alias import ASTObject

from .node import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    DeclarationStatement,
    Expression,
    ExpressionStatement,
    FloatLiteral,
    ForAllStatement,
    FunctionExpression,
    IdentifierExpression,
    Import,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
    ReturnStatement,
    SelectionStatement,
    Statement,
    TernaryExpression,
    TupleAccessExpression,
    TupleExpression,
    UnaryExpression,
)
from .span import Source, Span


def get_cls_name(obj: ASTObject) -> str:
    """Retrieve the class name of an object."""
    if not hasattr(obj, "get_key_name"):
        return obj.__class__.__name__

    return obj.get_key_name()


# def _get_visit_attrs(node: ASTObject) -> List[str]:
#     if isinstance(node, ASTNode):
#         return node.get_visit_attrs()
#     elif isinstance(node, IRDataType):
#         return ["primitive_data_type"]
#     elif isinstance(node, IRIndexType):
#         return ["lower_bound", "upper_bound", "stride"]
#     elif isinstance(node, IRNumericalType):
#         return ["data_type", "shape"]
#     else:
#         return []


class BasePass(ABC):
    """Abstract visitor pattern AST nodes and structures."""

    def __call__(self, node: Any) -> Any:
        return self.visit(node)

    def visit(self, node: Any) -> Any:
        """Visit a node or structure based on its class name.

        Args:
            node (Any): AST node or structure to visit.

        Returns:
            Any: Result of visiting the node.

        """
        name = f"visit_{get_cls_name(node)}"
        method: Callable[[Any], Any] = getattr(self, name, self.default)

        return method(node)

    def default(self, node: Any) -> Any:
        """Visit a node that is not supported.

        Args:
            node (Any): AST node or structure to visit.

        Returns:
            Any: Result of visiting the node.

        Raises:
            NotImplementedError: If the node is not supported.

        """
        raise NotImplementedError(f'Node "{type(node)}" is not supported.')


class Visitor(BasePass):
    """AST node visitor."""

    def visit(self, node: Union[ASTObject, List[ASTObject]]) -> None:
        if isinstance(node, list):
            self.visit_list(node)
        else:
            super().visit(node)

    def visit_Module(self, node: Module) -> None:
        """Visit a module node.

        Args:
            node (Module): Module node to visit.

        """
        self.visit(node.name)
        self.visit(node.statements)

    def visit_Import(self, node: Import) -> None:
        """Visit an import node.

        Args:
            node (Import): Import node to visit.

        """
        self.visit(node.name)

    def visit_Operation(self, node: Operation) -> None:
        """Visit an operation node.

        Args:
            node (Operation): Operation node to visit.

        """
        self.visit(node.name)
        self.visit(node.args)
        self.visit(node.return_type)
        self.visit(node.body)

    def visit_Procedure(self, node: Procedure) -> None:
        """Visit a procedure node.

        Args:
            node (Procedure): Procedure node to visit.

        """
        self.visit(node.name)
        self.visit(node.args)
        self.visit(node.body)

    def visit_Argument(self, node: Argument) -> None:
        """Visit an argument node.

        Args:
            node (Argument): Argument node to visit.

        """
        self.visit(node.qualified_type)
        if node.name is not None:
            self.visit(node.name)

    def visit_DeclarationStatement(self, node: DeclarationStatement) -> None:
        """Visit a declaration statement node.

        Args:
            node (DeclarationStatement): Declaration statement node to visit.

        """
        self.visit(node.variable_name)
        self.visit(node.variable_type)
        if node.expression is not None:
            self.visit(node.expression)

    def visit_ExpressionStatement(self, node: ExpressionStatement) -> None:
        """Visit an expression statement node.

        Args:
            node (ExpressionStatement): Expression statement node to visit.

        """
        if node.left is not None:
            self.visit(node.left)
        self.visit(node.right)

    def visit_SelectionStatement(self, node: SelectionStatement) -> None:
        """Visit a selection statement node.

        Args:
            node (SelectionStatement): Selection statement node to visit.

        """
        self.visit(node.condition)
        self.visit(node.true_body)
        self.visit(node.false_body)

    def visit_ForAllStatement(self, node: ForAllStatement) -> None:
        """Visit a for-all statement node.

        Args:
            node (ForAllStatement): For-all statement node to visit.

        """
        self.visit(node.index)
        self.visit(node.body)

    def visit_ReturnStatement(self, node: ReturnStatement) -> None:
        """Visit a return statement node.

        Args:
            node (ReturnStatement): Return statement node to visit.

        """
        self.visit(node.expression)

    def visit_UnaryExpression(self, node: UnaryExpression) -> None:
        """Visit a unary expression node.

        Args:
            node (UnaryExpression): Unary expression node to visit.

        """
        self.visit(node.expression)

    def visit_BinaryExpression(self, node: BinaryExpression) -> None:
        """Visit a binary expression node.

        Args:
            node (BinaryExpression): Binary expression node to visit.

        """
        self.visit(node.left)
        self.visit(node.right)

    def visit_TernaryExpression(self, node: TernaryExpression) -> None:
        """Visit a ternary expression node.

        Args:
            node (TernaryExpression): Ternary expression node to visit.

        """
        self.visit(node.condition)
        self.visit(node.true)
        self.visit(node.false)

    def visit_FunctionExpression(self, node: FunctionExpression) -> None:
        """Visit a function expression node.

        Args:
            node (FunctionExpression): Function expression node to visit.

        """
        self.visit(node.function)
        self.visit(node.template_types)
        self.visit(node.indices)
        self.visit(node.args)

    def visit_ArrayAccessExpression(self, node: ArrayAccessExpression) -> None:
        """Visit an array access expression node.

        Args:
            node (ArrayAccessExpression): Array access expression node to visit.

        """
        self.visit(node.array_expression)
        self.visit(node.indices)

    def visit_TupleExpression(self, node: TupleExpression) -> None:
        """Visit a tuple expression node.

        Args:
            node (TupleExpression): Tuple expression node to visit.

        """
        self.visit(node.expressions)

    def visit_TupleAccessExpression(self, node: TupleAccessExpression) -> None:
        """Visit a tuple access expression node.

        Args:
            node (TupleAccessExpression): Tuple access expression node to visit.

        """
        self.visit(node.tuple_expression)
        self.visit(node.element_index)

    def visit_IdentifierExpression(self, node: IdentifierExpression) -> None:
        """Visit an identifier expression node.

        Args:
            node (IdentifierExpression): Identifier expression node to visit.

        """
        self.visit(node.identifier)

    def visit_IntLiteral(self, node: IntLiteral) -> None:
        """Visit an integer literal node.

        Args:
            node (IntLiteral): Integer literal node to visit.

        """

    def visit_FloatLiteral(self, node: FloatLiteral) -> None:
        """Visit a float literal node.

        Args:
            node (FloatLiteral): Float literal node to visit.

        """

    def visit_QualifiedType(self, node: QualifiedType) -> None:
        """Visit a qualified type node.

        Args:
            node (QualifiedType): Qualified type node to visit.

        """
        self.visit(node.base_type)
        self.visit(node.type_qualifier)

    def visit_NumericalType(self, numerical_type: IRNumericalType) -> None:
        """Visit a numerical type.

        Args:
            numerical_type (IRNumericalType): Numerical type to visit.

        """
        self.visit(numerical_type.data_type)
        self.visit(numerical_type.shape)

    def visit_IndexType(self, index_type: IRIndexType) -> None:
        """Visit an index type.

        Args:
            index_type (IRIndexType): Index type to visit.

        """
        self.visit(index_type.lower_bound)
        self.visit(index_type.upper_bound)
        if index_type.stride is not None:
            self.visit(index_type.stride)

    def visit_DataType(self, data_type: IRDataType) -> None:
        """Visit a data type.

        Args:
            data_type (IRDataType): Data type to visit.

        """
        self.visit(data_type.primitive_data_type)

    def visit_TypeQualifier(self, type_qualifier: IRTypeQualifier) -> None:
        """Visit a type qualifier.

        Args:
            type_qualifier (IRTypeQualifier): Type qualifier to visit.

        """

    def visit_PrimitiveDataType(self, primitive: IRPrimitiveDataType) -> None:
        """Visit a primitive data type.

        Args:
            primitive (IRPrimitiveDataType): Primitive data type to visit.

        """

    def visit_Identifier(self, identifier: IRIdentifier) -> None:
        """Visit an identifier.

        Args:
            identifier (IRIdentifier): Identifier to visit.

        """

    def visit_list(self, nodes: list[ASTObject]) -> None:
        """Visit a list of nodes or structures.

        Args:
            nodes (List[ASTObject]): Nodes or structures to visit.

        """
        for node in nodes:
            self.visit(node)

    def visit_Span(self, span: Span) -> None:
        """Visit a span.

        Args:
            span (Span): Span to visit.

        """
        self.visit(span.source)

    def visit_Source(self, source: Source) -> None:
        """Visit a source.

        Args:
            source (Source): Source to visit.

        """


# class Listener(BasePass):
#     """AST node listener.

#     Listener is a visitor that does not control how to visit child nodes.
#     """

#     def __call__(self, node: Union[ASTObject, List[ASTObject]]) -> None:
#         return self.visit(node)

#     def visit(self, node: Union[ASTObject, List[ASTObject]]) -> None:
#         name = f"visit_{get_cls_name(node)}"
#         method: Callable[[Any], Any] = getattr(self, name, self.default)

#         return method(node)

#     # def _dispatch(self, node: Union[ASTObject, List[ASTObject]]) -> None:
#     #     node_stack = Stack()
#     #     node_stack.push(node)

#     #     while len(node_stack) != 0:
#     #         current_node = node_stack.pop()
#     #         nodes_to_visit = _get_visit_attrs(current_node)
#     #         for n in nodes_to_visit:
#     #             node_stack.push(getattr(current_node, n))

#     # def _get_visit_methods(
#     #     self,
#     #     node: Union[ASTObject, List[ASTObject]]
#     # ) -> Tuple[List[Callable[[], None]], List[Callable[[], None]]]:
#     #     if isinstance(node, ASTObject):
#     #         return self._get_ast_object_visit_methods(node)
#     #     else:
#     #         visit_enter_methods: List[Callable[[], None]] = []
#     #         visit_exit_methods: List[Callable[[], None]] = []
#     #         for n in node:
#     #             enter, exit = self._get_ast_object_visit_methods(n)
#     #             visit_enter_methods.extend(enter)
#     #             visit_exit_methods.extend(exit)

#     # def _get_ast_object_visit_methods(
#     #     self,
#     #     node: ASTObject
#     # ) -> Tuple[List[Callable[[], None]], List[Callable[[], None]]]:
#     #     visit_attrs = _get_visit_attrs(node)
#     #     visit_enter_methods: List[Callable[[], None]] = []
#     #     visit_exit_methods: List[Callable[[], None]] = []
#     #     for attr in visit_attrs:
#     #         method_enter = f"enter_{get_cls_name(attr)}"
#     #         method_exit = f"exit_{get_cls_name(attr)}"
#     #         visit_enter_methods.append(partial(getattr(self, method_enter), attr))
#     #         visit_exit_methods.append(partial(getattr(self, method_exit), attr))
#     #     return visit_enter_methods, visit_exit_methods

#     def default(self, node: Union[ASTObject, Sequence[ASTObject]]) -> None:
#         if isinstance(node, list):
#             return self.enter_sequence(node)

#         return super().default(node)

#     def enter_Module(self, node: Module) -> None: ...
#     def exit_Module(self, node: Module) -> None: ...

#     def enter_Operation(self, node: Operation) -> None: ...
#     def exit_Operation(self, node: Operation) -> None: ...

#     def enter_Procedure(self, node: Procedure) -> None: ...
#     def exit_Procedure(self, node: Procedure) -> None: ...

#     def enter_Argument(self, node: Argument) -> None: ...
#     def exit_Argument(self, node: Argument) -> None: ...

#     def enter_DeclarationStatement(self, node: DeclarationStatement) -> None: ...
#     def exit_DeclarationStatement(self, node: DeclarationStatement) -> None: ...

#     def enter_ExpressionStatement(self, node: ExpressionStatement) -> None: ...
#     def exit_ExpressionStatement(self, node: ExpressionStatement) -> None: ...

#     def enter_SelectionStatement(self, node: SelectionStatement) -> None: ...
#     def exit_SelectionStatement(self, node: SelectionStatement) -> None: ...

#     def enter_ForAllStatement(self, node: ForAllStatement) -> None: ...
#     def exit_ForAllStatement(self, node: ForAllStatement) -> None: ...

#     def enter_ReturnStatement(self, node: ReturnStatement) -> None: ...
#     def exit_ReturnStatement(self, node: ReturnStatement) -> None: ...

#     def enter_UnaryExpression(self, node: UnaryExpression) -> None: ...
#     def exit_UnaryExpression(self, node: UnaryExpression) -> None: ...

#     def enter_BinaryExpression(self, node: BinaryExpression) -> None: ...
#     def exit_BinaryExpression(self, node: BinaryExpression) -> None: ...

#     def enter_TernaryExpression(self, node: TernaryExpression) -> None: ...
#     def exit_TernaryExpression(self, node: TernaryExpression) -> None: ...

#     def enter_FunctionExpression(self, node: FunctionExpression) -> None: ...
#     def exit_FunctionExpression(self, node: FunctionExpression) -> None: ...

#     def enter_ArrayAccessExpression(self, node: ArrayAccessExpression) -> None: ...
#     def exit_ArrayAccessExpression(self, node: ArrayAccessExpression) -> None: ...

#     def enter_TupleExpression(self, node: TupleExpression) -> None: ...
#     def exit_TupleExpression(self, node: TupleExpression) -> None: ...

#     def enter_TupleAccessExpression(self, node: TupleAccessExpression) -> None: ...
#     def exit_TupleAccessExpression(self, node: TupleAccessExpression) -> None: ...

#     def enter_IdentifierExpression(self, node: IdentifierExpression) -> None: ...
#     def exit_IdentifierExpression(self, node: IdentifierExpression) -> None: ...

#     def enter_IntLiteral(self, node: IntLiteral) -> None: ...
#     def exit_IntLiteral(self, node: IntLiteral) -> None: ...

#     def enter_FloatLiteral(self, node: FloatLiteral) -> None: ...
#     def exit_FloatLiteral(self, node: FloatLiteral) -> None: ...

#     def enter_QualifiedType(self, node: QualifiedType) -> None: ...
#     def exit_QualifiedType(self, node: QualifiedType) -> None: ...

#     def enter_DataType(self, node: IRDataType) -> None: ...
#     def exit_DataType(self, node: IRDataType) -> None: ...

#     def enter_NumericalType(self, numerical_type: IRNumericalType) -> None: ...
#     def exit_NumericalType(self, numerical_type: IRNumericalType) -> None: ...

#     def enter_IndexType(self, index_type: IRIndexType) -> None: ...
#     def exit_IndexType(self, index_type: IRIndexType) -> None: ...

#     def enter_Identifier(self, identifier: IRIdentifier) -> None: ...
#     def exit_Identifier(self, identifier: IRIdentifier) -> None: ...

#     def enter_sequence(self, nodes: Sequence[ASTObject]) -> None: ...
#     def exit_sequence(self, nodes: Sequence[ASTObject]) -> None: ...

#     def enter_Span(self, span: Span) -> None: ...
#     def exit_Span(self, span: Span) -> None: ...

#     def enter_Source(self, source: Source) -> None: ...
#     def exit_Source(self, source: Source) -> None: ...


class Transformer(BasePass):
    """AST node transformer."""

    def visit_Module(self, node: Module) -> Module:
        span: Span = self.visit_Span(node.span)
        new_name = self.visit_Identifier(node.name)
        new_statements = self.visit_Statements(node.statements)

        return Module(span=span, name=new_name, statements=new_statements)

    def visit_Statements(self, nodes: List[Statement]) -> List[Statement]:
        return [self.visit_Statement(node) for node in nodes]

    def visit_Statement(self, node: Statement) -> Statement:
        if isinstance(node, Import):
            return self.visit_Import(node)
        elif isinstance(node, Operation):
            return self.visit_Operation(node)
        elif isinstance(node, Procedure):
            return self.visit_Procedure(node)
        elif isinstance(node, DeclarationStatement):
            return self.visit_DeclarationStatement(node)
        elif isinstance(node, ExpressionStatement):
            return self.visit_ExpressionStatement(node)
        elif isinstance(node, SelectionStatement):
            return self.visit_SelectionStatement(node)
        elif isinstance(node, ForAllStatement):
            return self.visit_ForAllStatement(node)
        elif isinstance(node, ReturnStatement):
            return self.visit_ReturnStatement(node)
        else:
            raise NotImplementedError(f'Node "{type(node)}" is not supported.')

    def visit_Import(self, node: Import) -> Import:
        new_name = self.visit_Identifier(node.name)
        span: Span = self.visit_Span(node.span)

        return Import(span=span, name=new_name)

    def visit_Operation(self, node: Operation) -> Operation:
        span: Span = self.visit_Span(node.span)
        new_name = self.visit_Identifier(node.name)
        new_args = self.visit_Arguments(node.args)
        new_return_type = self.visit_QualifiedType(node.return_type)
        new_body: List[Statement] = self.visit_Statements(node.body)

        return Operation(
            span=span,
            name=new_name,
            args=new_args,
            return_type=new_return_type,
            body=new_body,
        )

    def visit_Procedure(self, node: Procedure) -> Procedure:
        span: Span = self.visit_Span(node.span)
        new_name = self.visit_Identifier(node.name)
        new_args = self.visit_Arguments(node.args)
        new_body = self.visit_Statements(node.body)

        return Procedure(span=span, name=new_name, args=new_args, body=new_body)

    def visit_Arguments(self, nodes: List[Argument]) -> List[Argument]:
        return [self.visit_Argument(node) for node in nodes]

    def visit_Argument(self, node: Argument) -> Argument:
        span: Span = self.visit_Span(node.span)
        new_qualified_type = self.visit_QualifiedType(node.qualified_type)
        if node.name is not None:
            new_name = self.visit_Identifier(node.name)
        else:
            new_name = None

        return Argument(span=span, qualified_type=new_qualified_type, name=new_name)

    def visit_DeclarationStatement(
        self, node: DeclarationStatement
    ) -> DeclarationStatement:
        span: Span = self.visit_Span(node.span)
        new_variable_name = self.visit_Identifier(node.variable_name)
        new_variable_type = self.visit_QualifiedType(node.variable_type)
        if node.expression is not None:
            new_expression = self.visit_Expression(node.expression)
        else:
            new_expression = None

        return DeclarationStatement(
            span=span,
            variable_name=new_variable_name,
            variable_type=new_variable_type,
            expression=new_expression,
        )

    def visit_ExpressionStatement(
        self, node: ExpressionStatement
    ) -> ExpressionStatement:
        span: Span = self.visit_Span(node.span)
        if node.left is not None:
            new_left = self.visit_Expression(node.left)
        else:
            new_left = None
        new_right = self.visit(node.right)

        return ExpressionStatement(span=span, left=new_left, right=new_right)

    def visit_SelectionStatement(self, node: SelectionStatement) -> SelectionStatement:
        span: Span = self.visit_Span(node.span)
        new_condition = self.visit_Expression(node.condition)
        new_true_body = self.visit_Statements(node.true_body)
        new_false_body = self.visit_Statements(node.false_body)

        return SelectionStatement(
            span=span,
            condition=new_condition,
            true_body=new_true_body,
            false_body=new_false_body,
        )

    def visit_ForAllStatement(self, node: ForAllStatement) -> ForAllStatement:
        span: Span = self.visit_Span(node.span)
        new_index = self.visit_Expression(node.index)
        new_body = self.visit_Statements(node.body)

        return ForAllStatement(span=span, index=new_index, body=new_body)

    def visit_ReturnStatement(self, node: ReturnStatement) -> ReturnStatement:
        span: Span = self.visit_Span(node.span)
        new_expression = self.visit(node.expression)

        return ReturnStatement(span=span, expression=new_expression)

    def visit_Expressions(self, nodes: Sequence[Expression]) -> Sequence[Expression]:
        return [self.visit_Expression(node) for node in nodes]

    def visit_Expression(self, node: Expression) -> Expression:
        if isinstance(node, UnaryExpression):
            return self.visit_UnaryExpression(node)
        elif isinstance(node, BinaryExpression):
            return self.visit_BinaryExpression(node)
        elif isinstance(node, TernaryExpression):
            return self.visit_TernaryExpression(node)
        elif isinstance(node, FunctionExpression):
            return self.visit_FunctionExpression(node)
        elif isinstance(node, ArrayAccessExpression):
            return self.visit_ArrayAccessExpression(node)
        elif isinstance(node, TupleExpression):
            return self.visit_TupleExpression(node)
        elif isinstance(node, TupleAccessExpression):
            return self.visit_TupleAccessExpression(node)
        elif isinstance(node, IdentifierExpression):
            return self.visit_IdentifierExpression(node)
        elif isinstance(node, IntLiteral):
            return self.visit_IntLiteral(node)
        elif isinstance(node, FloatLiteral):
            return self.visit_FloatLiteral(node)
        else:
            raise NotImplementedError(f'Node "{type(node)}" is not supported.')

    def visit_UnaryExpression(self, node: UnaryExpression) -> UnaryExpression:
        span: Span = self.visit_Span(node.span)
        new_expression = self.visit_Expression(node.expression)

        return UnaryExpression(
            span=span, operation=node.operation, expression=new_expression
        )

    def visit_BinaryExpression(self, node: BinaryExpression) -> BinaryExpression:
        span: Span = self.visit_Span(node.span)
        new_left = self.visit_Expression(node.left)
        new_right = self.visit_Expression(node.right)

        return BinaryExpression(
            span=span, operation=node.operation, left=new_left, right=new_right
        )

    def visit_TernaryExpression(self, node: TernaryExpression) -> TernaryExpression:
        span: Span = self.visit_Span(node.span)
        new_condition = self.visit_Expression(node.condition)
        new_true = self.visit_Expression(node.true)
        new_false = self.visit_Expression(node.false)

        return TernaryExpression(
            span=span, condition=new_condition, true=new_true, false=new_false
        )

    def visit_FunctionExpression(self, node: FunctionExpression) -> FunctionExpression:
        span: Span = self.visit_Span(node.span)
        new_function = self.visit_Expression(node.function)
        new_template_types = self.visit_Types(node.template_types)
        new_indices = self.visit_Expressions(node.indices)
        new_args = self.visit_Expressions(node.args)

        return FunctionExpression(
            span=span,
            function=new_function,
            template_types=new_template_types,
            indices=list(new_indices),
            args=list(new_args),
        )

    def visit_ArrayAccessExpression(
        self, node: ArrayAccessExpression
    ) -> ArrayAccessExpression:
        span: Span = self.visit_Span(node.span)
        new_array_expresssion = self.visit_Expression(node.array_expression)
        new_indices = list(self.visit_Expressions(node.indices))

        return ArrayAccessExpression(
            span=span, array_expression=new_array_expresssion, indices=new_indices
        )

    def visit_TupleExpression(self, node: TupleExpression) -> TupleExpression:
        span: Span = self.visit_Span(node.span)
        new_expressions = self.visit_Expressions(node.expressions)

        return TupleExpression(span=span, expressions=list(new_expressions))

    def visit_TupleAccessExpression(
        self, node: TupleAccessExpression
    ) -> TupleAccessExpression:
        span: Span = self.visit_Span(node.span)
        new_tuple_expression = self.visit(node.tuple_expression)
        new_element_index = self.visit_IntLiteral(node.element_index)

        return TupleAccessExpression(
            span=span,
            tuple_expression=new_tuple_expression,
            element_index=new_element_index,
        )

    def visit_IdentifierExpression(
        self, node: IdentifierExpression
    ) -> IdentifierExpression:
        span: Span = self.visit_Span(node.span)
        new_identifier = self.visit_Identifier(node.identifier)

        return IdentifierExpression(span=span, identifier=new_identifier)

    def visit_IntLiteral(self, node: IntLiteral) -> IntLiteral:
        return copy(node)

    def visit_FloatLiteral(self, node: FloatLiteral) -> FloatLiteral:
        return copy(node)

    def visit_QualifiedType(self, node: QualifiedType) -> QualifiedType:
        span: Span = self.visit_Span(node.span)
        new_base_type = self.visit_Type(node.base_type)
        new_type_qualifier = self.visit_TypeQualifier(node.type_qualifier)

        return QualifiedType(
            span=span, base_type=new_base_type, type_qualifier=new_type_qualifier
        )

    def visit_Types(self, nodes: List[IRType]) -> List[IRType]:
        return [self.visit_Type(node) for node in nodes]

    def visit_Type(self, node: IRType) -> IRType:
        if isinstance(node, IRNumericalType):
            return self.visit_NumericalType(node)
        elif isinstance(node, IRIndexType):
            return self.visit_IndexType(node)
        else:
            raise NotImplementedError(f'Node "{type(node)}" is not supported.')

    def visit_NumericalType(self, numerical_type: IRNumericalType) -> IRNumericalType:
        new_data_type = self.visit_DataType(numerical_type.data_type)
        new_shape = [self.visit(j) for j in numerical_type.shape]

        return IRNumericalType(data_type=new_data_type, shape=new_shape)

    def visit_IndexType(self, index_type: IRIndexType) -> IRIndexType:
        new_lower_bound = self.visit(index_type.lower_bound)
        new_upper_bound = self.visit(index_type.upper_bound)
        if index_type.stride is not None:
            new_stride = self.visit(index_type.stride)
        else:
            new_stride = None

        return IRIndexType(
            lower_bound=new_lower_bound,
            upper_bound=new_upper_bound,
            stride=new_stride,
        )

    def visit_DataType(self, data_type: IRDataType) -> IRDataType:
        new_primitive_data_type = self.visit(data_type.primitive_data_type)

        return IRDataType(primitive_data_type=new_primitive_data_type)

    def visit_TypeQualifier(self, type_qualifier: IRTypeQualifier) -> IRTypeQualifier:
        return copy(type_qualifier)

    def visit_PrimitiveDataType(
        self, primitive: IRPrimitiveDataType
    ) -> IRPrimitiveDataType:
        return copy(primitive)

    def visit_Identifier(self, identifier: IRIdentifier) -> IRIdentifier:
        return copy(identifier)

    def visit_Span(self, span: Span) -> Span:
        # TODO: fix to copy everything
        # new_source = self.visit_Source(span.source)
        # return Span(source=new_source)
        return span

    def visit_Source(self, source: Source) -> Source:
        return copy(source)
