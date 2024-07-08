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
    Transformer: Visitor to modify AST nodes.

"""

from abc import ABC
from collections.abc import Callable, Sequence
from copy import copy
from typing import Any

from fhy.ir.identifier import Identifier as IRIdentifier
from fhy.ir.type import CoreDataType as IRCoreDataType
from fhy.ir.type import IndexType as IRIndexType
from fhy.ir.type import NumericalType as IRNumericalType
from fhy.ir.type import PrimitiveDataType as IRPrimitiveDataType
from fhy.ir.type import TemplateDataType as IRTemplateDataType
from fhy.ir.type import Type as IRType
from fhy.ir.type import TypeQualifier as IRTypeQualifier
from fhy.lang.ast.alias import ASTObject

from .node import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    ComplexLiteral,
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


def get_cls_name(obj: ASTObject | Sequence[ASTObject]) -> str:
    """Retrieve the class name of an object."""
    if not hasattr(obj, "get_key_name"):
        return obj.__class__.__name__

    return obj.get_key_name()


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

    def visit(self, node: ASTObject | Sequence[ASTObject]) -> None:
        if isinstance(node, list):
            self.visit_sequence(node)
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

    def visit_ComplexLiteral(self, node: ComplexLiteral) -> None:
        """Visit a complex literal node.

        Args:
            node (ComplexLiteral): Complex literal node to visit.

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
            numerical_type (ir.NumericalType): Numerical type to visit.

        """
        self.visit(numerical_type.data_type)
        self.visit(numerical_type.shape)

    def visit_IndexType(self, index_type: IRIndexType) -> None:
        """Visit an index type.

        Args:
            index_type (ir.IndexType): Index type to visit.

        """
        self.visit(index_type.lower_bound)
        self.visit(index_type.upper_bound)
        if index_type.stride is not None:
            self.visit(index_type.stride)

    def visit_PrimitiveDataType(self, node: IRPrimitiveDataType) -> None:
        """Visit a primitive data type.

        Args:
            node (ir.PrimitiveDataType): Data type to visit.

        """
        self.visit(node.primitive_data_type)

    def visit_TemplateDataType(self, node: IRTemplateDataType) -> None:
        """Visit a template data type.

        Args:
            node (ir.TemplateDataType): Template data type to visit.

        """
        self.visit(node.template_type)

    def visit_TypeQualifier(self, type_qualifier: IRTypeQualifier) -> None:
        """Visit a type qualifier.

        Args:
            type_qualifier (ir.TypeQualifier): Type qualifier to visit.

        """

    def visit_CoreDataType(self, primitive: IRCoreDataType) -> None:
        """Visit a primitive data type.

        Args:
            primitive (ir.CoreDataType): PrimitiveDataType data type to visit.

        """

    def visit_Identifier(self, identifier: IRIdentifier) -> None:
        """Visit an identifier.

        Args:
            identifier (ir.Identifier): Identifier to visit.

        """

    def visit_sequence(self, nodes: Sequence[ASTObject]) -> None:
        """Visit a list of nodes or structures.

        Args:
            nodes (list[ASTObject]): Nodes or structures to visit.

        """
        for node in nodes:
            self.visit(node)

    def visit_Span(self, span: Span | None) -> None:
        """Visit a span.

        Args:
            span (Span): Span to visit.

        """
        if span is not None and span.source is not None:
            self.visit(span.source)

    def visit_Source(self, source: Source) -> None:
        """Visit a source.

        Args:
            source (Source): Source to visit.

        """


class Transformer(BasePass):
    """AST node transformer."""

    def visit_sequence(self, nodes: Sequence[ASTObject]) -> list[ASTObject]:
        """Visit a list of nodes or structures.

        Args:
            nodes (list[ASTObject]): Nodes or structures to visit.

        """
        return [self.visit(node) for node in nodes]

    def visit_Module(self, node: Module) -> Module:
        """Transform a module node.

        Args:
            node (Module): Module node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_name: IRIdentifier = self.visit_Identifier(node.name)
        new_statements: list[Statement] = self.visit_sequence(node.statements)

        return Module(span=span, name=new_name, statements=new_statements)

    def visit_Import(self, node: Import) -> Import:
        """Transform an import node.

        Args:
            node (Import): Import node to transform.

        """
        new_name: IRIdentifier = self.visit_Identifier(node.name)
        span: Span | None = self.visit_Span(node.span)

        return Import(span=span, name=new_name)

    def visit_Operation(self, node: Operation) -> Operation:
        """Transform an operation node.

        Args:
            node (Operation): Operation node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_name: IRIdentifier = self.visit_Identifier(node.name)
        new_args: list[Argument] = self.visit_Arguments(node.args)
        new_return_type: QualifiedType = self.visit_QualifiedType(node.return_type)
        new_body: list[Statement] = self.visit_sequence(node.body)

        return Operation(
            span=span,
            name=new_name,
            args=new_args,
            return_type=new_return_type,
            body=new_body,
        )

    def visit_Procedure(self, node: Procedure) -> Procedure:
        """Transform a Procedure node.

        Args:
            node (Procedure): Procedure node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_name: IRIdentifier = self.visit_Identifier(node.name)
        new_args: list[Argument] = self.visit_Arguments(node.args)
        new_body: list[Statement] = self.visit_sequence(node.body)

        return Procedure(span=span, name=new_name, args=new_args, body=new_body)

    def visit_Arguments(self, nodes: list[Argument]) -> list[Argument]:
        """Transform a list of argument nodes.

        Args:
            nodes (list[Argument]): List of Argument nodes to transform.

        """
        return [self.visit_Argument(node) for node in nodes]

    def visit_Argument(self, node: Argument) -> Argument:
        """Transform an argument node.

        Args:
            node (Argument): Argument node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_qualified_type: QualifiedType = self.visit_QualifiedType(
            node.qualified_type
        )
        new_name: IRIdentifier | None
        if node.name is not None:
            new_name = self.visit_Identifier(node.name)
        else:
            new_name = None

        return Argument(span=span, qualified_type=new_qualified_type, name=new_name)

    def visit_DeclarationStatement(
        self, node: DeclarationStatement
    ) -> DeclarationStatement:
        """Transform a declaration statement node.

        Args:
            node (DeclarationStatement): DeclarationStatement node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_variable_name: IRIdentifier = self.visit_Identifier(node.variable_name)
        new_variable_type: QualifiedType = self.visit_QualifiedType(node.variable_type)
        new_expression: Expression | None
        if node.expression is not None:
            new_expression = self.visit(node.expression)
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
        """Transform an expression statement node.

        Args:
            node (ExpressionStatement): ExpressionStatement node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_left: Expression | None
        if node.left is not None:
            new_left = self.visit(node.left)
        else:
            new_left = None
        new_right = self.visit(node.right)

        return ExpressionStatement(span=span, left=new_left, right=new_right)

    def visit_SelectionStatement(self, node: SelectionStatement) -> SelectionStatement:
        """Transform a selection statement node.

        Args:
            node (SelectionStatement): SelectionStatement node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_condition: Expression = self.visit(node.condition)
        new_true_body: list[Statement] = self.visit_sequence(node.true_body)
        new_false_body: list[Statement] = self.visit_sequence(node.false_body)

        return SelectionStatement(
            span=span,
            condition=new_condition,
            true_body=new_true_body,
            false_body=new_false_body,
        )

    def visit_ForAllStatement(self, node: ForAllStatement) -> ForAllStatement:
        """Transform an iteration statement node.

        Args:
            node (ForAllStatement): ForAllStatement node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_index: Expression = self.visit(node.index)
        new_body: list[Statement] = self.visit_sequence(node.body)

        return ForAllStatement(span=span, index=new_index, body=new_body)

    def visit_ReturnStatement(self, node: ReturnStatement) -> ReturnStatement:
        """Transform a return statement node.

        Args:
            node (ReturnStatement): ReturnStatement node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_expression: Expression = self.visit(node.expression)

        return ReturnStatement(span=span, expression=new_expression)

    def visit_UnaryExpression(self, node: UnaryExpression) -> UnaryExpression:
        """Transform a unary expression node.

        Args:
            node (UnaryExpression): UnaryExpression node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_expression: Expression = self.visit(node.expression)

        return UnaryExpression(
            span=span, operation=node.operation, expression=new_expression
        )

    def visit_BinaryExpression(self, node: BinaryExpression) -> BinaryExpression:
        """Transform a binary expression node.

        Args:
            node (BinaryExpression): BinaryExpression node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_left: Expression = self.visit(node.left)
        new_right: Expression = self.visit(node.right)

        return BinaryExpression(
            span=span, operation=node.operation, left=new_left, right=new_right
        )

    def visit_TernaryExpression(self, node: TernaryExpression) -> TernaryExpression:
        """Transform a ternary expression node.

        Args:
            node (TernaryExpression): TernaryExpression node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_condition: Expression = self.visit(node.condition)
        new_true: Expression = self.visit(node.true)
        new_false: Expression = self.visit(node.false)

        return TernaryExpression(
            span=span, condition=new_condition, true=new_true, false=new_false
        )

    def visit_FunctionExpression(self, node: FunctionExpression) -> FunctionExpression:
        """Transform a function expression node.

        Args:
            node (FunctionExpression): FunctionExpression node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_function: Expression = self.visit(node.function)
        new_template_types: list[IRType] = self.visit_Types(node.template_types)
        new_indices: list[Expression] = self.visit_sequence(node.indices)
        new_args: list[Expression] = self.visit_sequence(node.args)

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
        """Transform an array access expression node.

        Args:
            node (ArrayAccessExpression): ArrayAccessExpression node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_array_expresssion: Expression = self.visit(node.array_expression)
        new_indices: list[Expression] = self.visit_sequence(node.indices)

        return ArrayAccessExpression(
            span=span, array_expression=new_array_expresssion, indices=new_indices
        )

    def visit_TupleExpression(self, node: TupleExpression) -> TupleExpression:
        """Transform a tuple expression node.

        Args:
            node (TupleExpression): TupleExpression node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_expressions: list[Expression] = self.visit_sequence(node.expressions)

        return TupleExpression(span=span, expressions=list(new_expressions))

    def visit_TupleAccessExpression(
        self, node: TupleAccessExpression
    ) -> TupleAccessExpression:
        """Transform a tuple access expression node.

        Args:
            node (TupleAccessExpression): TupleAccessExpression node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_tuple_expression: Expression = self.visit(node.tuple_expression)
        new_element_index: IntLiteral = self.visit_IntLiteral(node.element_index)

        return TupleAccessExpression(
            span=span,
            tuple_expression=new_tuple_expression,
            element_index=new_element_index,
        )

    def visit_IdentifierExpression(
        self, node: IdentifierExpression
    ) -> IdentifierExpression:
        """Transform an identifier expression node.

        Args:
            node (IdentifierExpression): IdentifierExpression node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_identifier: IRIdentifier = self.visit_Identifier(node.identifier)

        return IdentifierExpression(span=span, identifier=new_identifier)

    def visit_IntLiteral(self, node: IntLiteral) -> IntLiteral:
        """Transform an int literal node.

        Args:
            node (IntLiteral): IntLiteral node to transform.

        """
        return copy(node)

    def visit_FloatLiteral(self, node: FloatLiteral) -> FloatLiteral:
        """Transform a float literal node.

        Args:
            node (FloatLiteral): FloatLiteral node to transform.

        """
        return copy(node)

    def visit_ComplexLiteral(self, node: ComplexLiteral) -> ComplexLiteral:
        """Transform a complex literal node.

        Args:
            node (ComplexLiteral): ComplexLiteral node to transform.

        """
        return copy(node)

    def visit_QualifiedType(self, node: QualifiedType) -> QualifiedType:
        """Transform a qualified type node.

        Args:
            node (QualifiedType): QualifiedType node to transform.

        """
        span: Span | None = self.visit_Span(node.span)
        new_base_type: IRType = self.visit(node.base_type)
        new_type_qualifier: IRTypeQualifier = self.visit_TypeQualifier(
            node.type_qualifier
        )

        return QualifiedType(
            span=span, base_type=new_base_type, type_qualifier=new_type_qualifier
        )

    def visit_Types(self, nodes: list[IRType]) -> list[IRType]:
        """Transform a list of type nodes.

        Args:
            nodes (list[ir.Type]): List of Type nodes to transform.

        """
        return [self.visit_Type(node) for node in nodes]

    def visit_Type(self, node: IRType) -> IRType:
        """Transform a type node.

        Args:
            node (IRType): Type node to transform.

        """
        if isinstance(node, IRNumericalType):
            return self.visit_NumericalType(node)
        elif isinstance(node, IRIndexType):
            return self.visit_IndexType(node)
        else:
            raise NotImplementedError(f'Node "{type(node)}" is not supported.')

    def visit_NumericalType(self, numerical_type: IRNumericalType) -> IRNumericalType:
        """Transform a numerical type node.

        Args:
            numerical_type (ir.NumericalType): NumericalType node to transform.

        """
        new_data_type = self.visit(numerical_type.data_type)
        new_shape = [self.visit(j) for j in numerical_type.shape]

        return IRNumericalType(data_type=new_data_type, shape=new_shape)

    def visit_IndexType(self, index_type: IRIndexType) -> IRIndexType:
        """Transform a numerical type node.

        Args:
            index_type (ir.IndexType): IndexType node to transform.

        """
        new_lower_bound = self.visit(index_type.lower_bound)
        new_upper_bound = self.visit(index_type.upper_bound)
        new_stride: Expression | None
        if index_type.stride is not None:
            new_stride = self.visit(index_type.stride)
        else:
            new_stride = None

        return IRIndexType(
            lower_bound=new_lower_bound,
            upper_bound=new_upper_bound,
            stride=new_stride,
        )

    def visit_PrimitiveDataType(self, node: IRPrimitiveDataType) -> IRPrimitiveDataType:
        """Transform a primitive type node.

        Args:
            node (ir.PrimitiveDataType): PrimitiveDataType data type node to transform.

        """
        new_primitive_data_type: IRCoreDataType = self.visit_CoreDataType(
            node.primitive_data_type
        )

        return IRPrimitiveDataType(data_type=new_primitive_data_type)

    def visit_TemplateDataType(self, node: IRTemplateDataType) -> IRTemplateDataType:
        """Transform a template data type node.

        Args:
            node (ir.TemplateDataType): Template data type node to transform.

        """
        new_primitive_data_type = self.visit(node.template_type)

        return IRTemplateDataType(data_type=new_primitive_data_type)

    def visit_TypeQualifier(self, type_qualifier: IRTypeQualifier) -> IRTypeQualifier:
        """Transform a type qualifier node.

        Args:
            type_qualifier (ir.TypeQualifier): TypeQualifier node to transform.

        """
        return copy(type_qualifier)

    def visit_CoreDataType(self, primitive: IRCoreDataType) -> IRCoreDataType:
        """Transform a primitive data type node.

        Args:
            primitive (ir.CoreDataType): CoreDataType node to transform.

        """
        return copy(primitive)

    def visit_Identifier(self, identifier: IRIdentifier) -> IRIdentifier:
        """Transform an identifier node.

        Args:
            identifier (ir.Identifier): Identifier node to transform.

        """
        return copy(identifier)

    def _visit_span(self, span: Span) -> Span:
        new_source: Source | None = self.visit_Source(span.source)

        return Span(
            source=new_source,
            start_line=span.line.start,
            end_line=span.line.stop,
            start_column=span.column.start,
            end_column=span.column.stop,
        )

    def visit_Span(self, span: Span | None) -> Span | None:
        """Transform a span node.

        Args:
            span (Span, optional): Span node to transform.

        """
        return span and self._visit_span(span)

    def visit_Source(self, source: Source | None) -> Source | None:
        """Transform a source node.

        Args:
            source (Source, optional): Source node to transform.

        """
        return source and copy(source)
