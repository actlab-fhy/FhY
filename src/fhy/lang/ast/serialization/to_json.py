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

"""Conversion (serialization) of AST nodes to and from json format.

We construct an intermediate data format, called AlmostJson, which serves as a data
container and node identifier to prepare conversion of AST nodes both to and from the
json format. The AlmostJson class, can recursively convert all child leaf nodes into a
dictionary object in preparation of json serialization. Conversely, data contained by
the AlmostJson class can be used to reconstruct AST Nodes.

The classes, ASTtoJSON and JSONtoAST, both employ a visitor pattern, and are the primary
drivers behind the respective transformations. Each class has a function (mimicking the
json library api) to dump and load, as the primary entry point of use.

Primary API:
    load: Convert json string into AST nodes.
    dump: Convert AST nodes into json string.

Core Classes:
    ASTtoJSON: ASTNode visitor constructing AlmostJson nodes.
    JSONtoAST: AlmostJson node visitor constructing ASTNodes.

"""

import json
from typing import Any, Callable, List, Optional, Sequence, Union

from fhy import ir
from fhy.lang import ast
from fhy.lang.ast import visitor
from fhy.lang.ast.alias import ASTObject
from fhy.lang.ast.span import Source, Span


def convert(value: object) -> Any:
    """Recursively Converts objects into dictionary records."""
    if isinstance(value, AlmostJson):
        return value.data()

    elif isinstance(value, list):
        return [convert(child) for child in value]

    else:
        return value


class AlmostJson:
    """Consistent Data Structure Format for JSON Preparations."""

    cls_name: str
    attributes: dict

    def __init__(self, cls_name: str, attributes: dict) -> None:
        self.cls_name = cls_name
        self.attributes = attributes

    def data(self) -> dict:
        """Return class attributes into a dictionary record, recursively."""
        values = {k: convert(v) for k, v in self.attributes.items()}
        return dict(cls_name=self.cls_name, attributes=values)

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, AlmostJson)
            and self.cls_name == value.cls_name
            and self.attributes == value.attributes
        )


JSONObject = Union[AlmostJson, Sequence[AlmostJson]]


class ASTtoJSON(visitor.BasePass):
    """Convert an AST Node into a json object."""

    def visit_Module(self, node: ast.Module) -> AlmostJson:
        statements: List[AlmostJson] = self.visit_sequence(node.statements)

        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                statements=statements,
            ),
        )

        return obj

    def visit_Operation(self, node: ast.Operation) -> AlmostJson:
        templates: List[AlmostJson] = self.visit_sequence(node.templates)
        args: List[AlmostJson] = self.visit_sequence(node.args)
        ret_type: AlmostJson = self.visit(node.return_type)
        body: List[AlmostJson] = self.visit_sequence(node.body)
        name: AlmostJson = self.visit_Identifier(node.name)
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                templates=templates,
                name=name,
                args=args,
                return_type=ret_type,
                body=body,
            ),
        )

        return obj

    def visit_Procedure(self, node: ast.Procedure) -> AlmostJson:
        templates: List[AlmostJson] = self.visit_sequence(node.templates)
        args: List[AlmostJson] = self.visit_sequence(node.args)
        body: List[AlmostJson] = self.visit_sequence(node.body)
        name: AlmostJson = self.visit_Identifier(node.name)

        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                templates=templates,
                name=name,
                args=args,
                body=body,
            ),
        )

        return obj

    def visit_Argument(self, node: ast.Argument) -> AlmostJson:
        qtype: AlmostJson = self.visit_QualifiedType(node.qualified_type)
        name: Optional[AlmostJson] = (
            self.visit_Identifier(node.name) if node.name is not None else None
        )
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span), qualified_type=qtype, name=name
            ),
        )

        return obj

    def visit_DeclarationStatement(self, node: ast.DeclarationStatement) -> AlmostJson:
        varname: AlmostJson = self.visit_Identifier(node.variable_name)
        vartype: AlmostJson = self.visit_QualifiedType(node.variable_type)
        express: Optional[AlmostJson] = (
            self.visit(node.expression) if node.expression is not None else None
        )
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                variable_name=varname,
                variable_type=vartype,
                expression=express,
            ),
        )

        return obj

    def visit_ExpressionStatement(self, node: ast.ExpressionStatement) -> AlmostJson:
        left: Optional[AlmostJson] = (
            self.visit(node.left) if node.left is not None else None
        )
        right: AlmostJson = self.visit(node.right)
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                left=left,
                right=right,
            ),
        )

        return obj

    def visit_SelectionStatement(self, node: ast.SelectionStatement) -> AlmostJson:
        condition: AlmostJson = self.visit(node.condition)
        tbody: List[AlmostJson] = self.visit_sequence(node.true_body)
        fbody: List[AlmostJson] = self.visit_sequence(node.false_body)
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                condition=condition,
                true_body=tbody,
                false_body=fbody,
            ),
        )

        return obj

    def visit_ForAllStatement(self, node: ast.ForAllStatement) -> AlmostJson:
        index: AlmostJson = self.visit(node.index)
        body: List[AlmostJson] = self.visit_sequence(node.body)
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                index=index,
                body=body,
            ),
        )

        return obj

    def visit_ReturnStatement(self, node: ast.ReturnStatement) -> AlmostJson:
        express: AlmostJson = self.visit(node.expression)
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                expression=express,
            ),
        )

        return obj

    def visit_UnaryExpression(self, node: ast.UnaryExpression) -> AlmostJson:
        express: AlmostJson = self.visit(node.expression)
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                expression=express,
                operation=node.operation.value,
            ),
        )

        return obj

    def visit_BinaryExpression(self, node: ast.BinaryExpression) -> AlmostJson:
        left: AlmostJson = self.visit(node.left)
        right: AlmostJson = self.visit(node.right)

        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                left=left,
                operation=node.operation.value,
                right=right,
            ),
        )

        return obj

    def visit_TernaryExpression(self, node: ast.TernaryExpression) -> AlmostJson:
        condition: AlmostJson = self.visit(node.condition)
        true: AlmostJson = self.visit(node.true)
        false: AlmostJson = self.visit(node.false)

        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                condition=condition,
                true=true,
                false=false,
            ),
        )

        return obj

    def visit_FunctionExpression(self, node: ast.FunctionExpression) -> AlmostJson:
        function: AlmostJson = self.visit(node.function)
        template: List[AlmostJson] = self.visit_sequence(node.template_types)
        index: List[AlmostJson] = self.visit_sequence(node.indices)
        args: List[AlmostJson] = self.visit_sequence(node.args)
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                function=function,
                template_types=template,
                indices=index,
                args=args,
            ),
        )

        return obj

    def visit_ArrayAccessExpression(
        self, node: ast.ArrayAccessExpression
    ) -> AlmostJson:
        array: AlmostJson = self.visit(node.array_expression)
        index: List[AlmostJson] = self.visit_sequence(node.indices)
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                array_expression=array,
                indices=index,
            ),
        )

        return obj

    def visit_TupleExpression(self, node: ast.TupleExpression) -> AlmostJson:
        express: List[AlmostJson] = self.visit_sequence(node.expressions)
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                expressions=express,
            ),
        )

        return obj

    def visit_TupleAccessExpression(
        self, node: ast.TupleAccessExpression
    ) -> AlmostJson:
        _tuple: AlmostJson = self.visit(node.tuple_expression)
        element: AlmostJson = self.visit_IntLiteral(node.element_index)
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                tuple_expression=_tuple,
                element_index=element,
            ),
        )

        return obj

    def visit_IdentifierExpression(self, node: ast.IdentifierExpression) -> AlmostJson:
        identifier: AlmostJson = self.visit(node.identifier)

        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(span=self.visit_Span(node.span), identifier=identifier),
        )

        return obj

    def visit_IntLiteral(self, node: ast.IntLiteral) -> AlmostJson:
        return AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(span=self.visit_Span(node.span), value=node.value),
        )

    def visit_FloatLiteral(self, node: ast.FloatLiteral) -> AlmostJson:
        return AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(span=self.visit_Span(node.span), value=node.value),
        )

    def visit_ComplexLiteral(self, node: ast.ComplexLiteral) -> AlmostJson:
        # NOTE: Complex Values are not JSON Serializable. We must Separate the
        #       real and imaginary parts contained by a dictionary.
        result = dict(real=node.value.real, imag=node.value.imag)
        return AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(span=self.visit_Span(node.span), value=result),
        )

    def visit_DataType(self, node: ir.DataType) -> AlmostJson:
        return AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(primitive_data_type=node.primitive_data_type.value),
        )

    def visit_QualifiedType(self, node: ast.QualifiedType) -> AlmostJson:
        base: AlmostJson = self.visit(node.base_type)

        return AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                base_type=base,
                type_qualifier=node.type_qualifier.value,
            ),
        )

    def visit_NumericalType(self, numerical_type: ir.NumericalType) -> AlmostJson:
        dtype: AlmostJson = self.visit_DataType(numerical_type.data_type)
        shape: List[AlmostJson] = self.visit_sequence(numerical_type.shape)

        return AlmostJson(
            cls_name=visitor.get_cls_name(numerical_type),
            attributes=dict(data_type=dtype, shape=shape),
        )

    def visit_IndexType(self, index_type: ir.IndexType) -> AlmostJson:
        lower: AlmostJson = self.visit(index_type.lower_bound)
        upper: AlmostJson = self.visit(index_type.upper_bound)
        if index_type.stride is not None:
            stride: AlmostJson = self.visit(index_type.stride)
        else:
            stride = self.visit_IntLiteral(
                ast.IntLiteral(value=1, span=Span(0, 0, 0, 0))
            )

        return AlmostJson(
            cls_name=visitor.get_cls_name(index_type),
            attributes=dict(lower_bound=lower, upper_bound=upper, stride=stride),
        )

    def visit_TupleType(self, tuple_type: ir.TupleType) -> AlmostJson:
        types: List[AlmostJson] = self.visit_sequence(tuple_type._types)

        return AlmostJson(
            cls_name=visitor.get_cls_name(tuple_type),
            attributes=dict(types=types),
        )

    def visit_Identifier(self, identifier: ir.Identifier) -> AlmostJson:
        return AlmostJson(
            cls_name=visitor.get_cls_name(identifier),
            attributes=dict(name_hint=identifier.name_hint, _id=identifier.id),
        )

    def visit_sequence(self, nodes: Sequence[ASTObject]) -> List[AlmostJson]:
        return [self.visit(node) for node in nodes]

    def visit_Span(self, span: Span) -> AlmostJson:
        if span.source is not None:
            source: AlmostJson = self.visit_Source(span.source)
        else:
            source = span.source

        return AlmostJson(
            cls_name=visitor.get_cls_name(span),
            attributes=dict(
                start_line=span.line.start,
                end_line=span.line.stop,
                start_column=span.column.start,
                end_column=span.column.stop,
                source=source,
            ),
        )

    def visit_Source(self, source: Source) -> AlmostJson:
        return AlmostJson(
            cls_name=visitor.get_cls_name(source),
            attributes=dict(namespace=source.namespace),
        )


class JSONtoAST(visitor.BasePass):
    """Converts a JSON object into AST Nodes."""

    def visit(self, node: Optional[JSONObject]) -> Any:
        if isinstance(node, list):
            return self.visit_sequence(node)

        elif not isinstance(node, AlmostJson):
            return self.default(node)

        name = f"visit_{node.cls_name}"
        method: Callable[[JSONObject], ASTObject]
        method = getattr(self, name, self.default)

        return method(node)

    def visit_Module(self, node: Optional[AlmostJson]) -> ast.Module:
        if node is None:
            raise ValueError("Invalid Module")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        statements: List[ast.Statement] = self.visit_sequence(values.get("statements"))

        return ast.Module(span=span, statements=statements)

    def visit_Operation(self, node: Optional[AlmostJson]) -> ast.Operation:
        if node is None:
            raise ValueError("Invalid Operation")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        templates: List[ir.Identifier] = self.visit_sequence(values.get("templates"))
        args: List[ast.Argument] = self.visit_sequence(values.get("args"))
        body: List[ast.Statement] = self.visit_sequence(values.get("body"))
        name: ir.Identifier = self.visit_Identifier(values.get("name"))
        ret_type: ast.QualifiedType = self.visit_QualifiedType(
            values.get("return_type")
        )

        return ast.Operation(
            span=span,
            name=name,
            templates=templates,
            args=args,
            body=body,
            return_type=ret_type,
        )

    def visit_Procedure(self, node: Optional[AlmostJson]) -> ast.Procedure:
        if node is None:
            raise ValueError("Invalid Procedure")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        templates: List[ir.Identifier] = self.visit_sequence(values.get("templates"))
        args: List[ast.Argument] = self.visit_sequence(values.get("args"))
        body: List[ast.Statement] = self.visit_sequence(values.get("body"))
        name: ir.Identifier = self.visit_Identifier(values.get("name"))

        return ast.Procedure(
            span=span, name=name, templates=templates, args=args, body=body
        )

    def visit_Argument(self, node: Optional[AlmostJson]) -> ast.Argument:
        if node is None:
            raise ValueError("Invalid Argument")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        qtype: ast.QualifiedType = self.visit_QualifiedType(
            values.get("qualified_type")
        )
        name: ir.Identifier = self.visit_Identifier(values.get("name"))

        return ast.Argument(span=span, name=name, qualified_type=qtype)

    def visit_DeclarationStatement(
        self, node: Optional[AlmostJson]
    ) -> ast.DeclarationStatement:
        if node is None:
            raise ValueError("Invalid DeclarationStatement")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        varname: ir.Identifier = self.visit_Identifier(values.get("variable_name"))
        vartype: ast.QualifiedType = self.visit_QualifiedType(
            values.get("variable_type")
        )
        if (_express := values.get("expression")) is not None:
            values["expression"] = self.visit(_express)
        express = values.get("expression")

        return ast.DeclarationStatement(
            span=span, variable_name=varname, variable_type=vartype, expression=express
        )

    def visit_ExpressionStatement(
        self, node: Optional[AlmostJson]
    ) -> ast.ExpressionStatement:
        if node is None:
            raise ValueError("Invalid ExpressionStatement")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        if (_left := values.get("left")) is not None:
            values["left"] = self.visit(_left)
        left: Optional[ast.Expression] = values.get("left")
        right: ast.Expression = self.visit(values.get("right"))

        return ast.ExpressionStatement(span=span, left=left, right=right)

    def visit_SelectionStatement(
        self, node: Optional[AlmostJson]
    ) -> ast.SelectionStatement:
        if node is None:
            raise ValueError("Invalid SelectionStatement")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        condition: ast.Expression = self.visit(values.get("condition"))
        tbody: List[ast.Statement] = self.visit_sequence(values.get("true_body"))
        fbody: List[ast.Statement] = self.visit_sequence(values.get("false_body"))

        return ast.SelectionStatement(
            span=span, condition=condition, true_body=tbody, false_body=fbody
        )

    def visit_ForAllStatement(self, node: Optional[AlmostJson]) -> ast.ForAllStatement:
        if node is None:
            raise ValueError("Invalid ForAllStatement")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        index: ast.Expression = self.visit(values.get("index"))
        body: List[ast.Statement] = self.visit_sequence(values.get("body"))

        return ast.ForAllStatement(span=span, index=index, body=body)

    def visit_ReturnStatement(self, node: Optional[AlmostJson]) -> ast.ReturnStatement:
        if node is None:
            raise ValueError("Invalid ReturnStatement")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        express: ast.Expression = self.visit(values.get("expression"))

        return ast.ReturnStatement(span=span, expression=express)

    def visit_UnaryExpression(self, node: Optional[AlmostJson]) -> ast.UnaryExpression:
        if node is None:
            raise ValueError("Invalid UnaryExpression")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        express: ast.Expression = self.visit(values.get("expression"))
        operator: ast.UnaryOperation = ast.UnaryOperation(str(values.get("operation")))

        return ast.UnaryExpression(span=span, operation=operator, expression=express)

    def visit_BinaryExpression(
        self, node: Optional[AlmostJson]
    ) -> ast.BinaryExpression:
        if node is None:
            raise ValueError("Invalid BinaryExpression")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        left: ast.Expression = self.visit(values.get("left"))
        right: ast.Expression = self.visit(values.get("right"))
        operator = ast.BinaryOperation(str(values.get("operation")))

        return ast.BinaryExpression(
            span=span, left=left, right=right, operation=operator
        )

    def visit_TernaryExpression(
        self, node: Optional[AlmostJson]
    ) -> ast.TernaryExpression:
        if node is None:
            raise ValueError("Invalid TernaryExpression")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        condition: ast.Expression = self.visit(values.get("condition"))
        true: ast.Expression = self.visit(values.get("true"))
        false: ast.Expression = self.visit(values.get("false"))

        return ast.TernaryExpression(
            span=span, condition=condition, true=true, false=false
        )

    def visit_FunctionExpression(
        self, node: Optional[AlmostJson]
    ) -> ast.FunctionExpression:
        if node is None:
            raise ValueError("Invalid FunctionExpression")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        function: ast.Expression = self.visit(values.get("function"))
        template: List[ir.Type] = self.visit_sequence(values.get("template_types"))
        index: List[ast.Expression] = self.visit_sequence(values.get("indices"))
        args: List[ast.Expression] = self.visit_sequence(values.get("args"))

        return ast.FunctionExpression(
            span=span,
            function=function,
            template_types=template,
            indices=index,
            args=args,
        )

    def visit_ArrayAccessExpression(
        self, node: Optional[AlmostJson]
    ) -> ast.ArrayAccessExpression:
        if node is None:
            raise ValueError("Invalid ArrayAccessExpression")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        array: ast.Expression = self.visit(values.get("array_expression"))
        index: List[ast.Expression] = self.visit_sequence(values.get("indices"))

        return ast.ArrayAccessExpression(
            span=span,
            array_expression=array,
            indices=index,
        )

    def visit_TupleExpression(self, node: Optional[AlmostJson]) -> ast.TupleExpression:
        if node is None:
            raise ValueError("Invalid TupleExpression")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        express: List[ast.Expression] = self.visit_sequence(values.get("expressions"))

        return ast.TupleExpression(span=span, expressions=express)

    def visit_TupleAccessExpression(
        self, node: Optional[AlmostJson]
    ) -> ast.TupleAccessExpression:
        if node is None:
            raise ValueError("Invalid TupleAccessExpression")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        _tuple: ast.Expression = self.visit(values.get("tuple_expression"))
        element: ast.IntLiteral = self.visit_IntLiteral(values.get("element_index"))

        return ast.TupleAccessExpression(
            span=span, tuple_expression=_tuple, element_index=element
        )

    def visit_IdentifierExpression(
        self, node: Optional[AlmostJson]
    ) -> ast.IdentifierExpression:
        if node is None:
            raise ValueError("Invalid IdentifierExpression")

        values: dict = node.attributes
        identifier: ir.Identifier = self.visit_Identifier(values.get("identifier"))
        span: Span = self.visit_Span(values.get("span"))

        return ast.IdentifierExpression(span=span, identifier=identifier)

    def visit_IntLiteral(self, node: Optional[AlmostJson]) -> ast.IntLiteral:
        if node is None:
            raise ValueError("Invalid IntLiteral")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        if (value := values.get("value")) is None:
            raise ValueError("Invalid IntLiteral Value")

        return ast.IntLiteral(span=span, value=value)

    def visit_FloatLiteral(self, node: Optional[AlmostJson]) -> ast.FloatLiteral:
        if node is None:
            raise ValueError("Invalid FloatLiteral")
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        if (value := values.get("value")) is None:
            raise ValueError("Invalid FloatLiteral Value")

        return ast.FloatLiteral(span=span, value=value)

    def visit_ComplexLiteral(self, node: Optional[AlmostJson]) -> ast.ComplexLiteral:
        if node is None:
            raise ValueError("Invalid ComplexLiteral")

        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        if (value := values.get("value")) is None:
            raise ValueError("Invalid ComplexLiteral Value")

        # Combine real and imaginary parts to construct the complex number
        result: complex = complex(real=value.get("real"), imag=value.get("imag"))

        return ast.ComplexLiteral(span=span, value=result)

    def visit_DataType(self, node: Optional[AlmostJson]) -> ir.DataType:
        if node is None:
            raise ValueError("Invalid DataType")

        primitive = ir.PrimitiveDataType(
            str(node.attributes.get("primitive_data_type"))
        )

        return ir.DataType(primitive_data_type=primitive)

    def visit_QualifiedType(self, node: Optional[AlmostJson]) -> ast.QualifiedType:
        if node is None:
            raise ValueError("Invalid QualifiedType")

        values: dict = node.attributes
        base: ir.Type = self.visit(values.get("base_type"))
        span: Span = self.visit_Span(values.get("span"))
        qtype = ir.TypeQualifier(str(values.get("type_qualifier")))

        return ast.QualifiedType(
            span=span,
            base_type=base,
            type_qualifier=qtype,
        )

    def visit_NumericalType(
        self, numerical_type: Optional[AlmostJson]
    ) -> ir.NumericalType:
        if numerical_type is None:
            raise ValueError("Invalid numerical_type")
        values: dict = numerical_type.attributes
        dtype: ir.DataType = self.visit_DataType(values.get("data_type"))
        shape: List[ir.Expression] = self.visit_sequence(values.get("shape"))

        return ir.NumericalType(data_type=dtype, shape=shape)

    def visit_IndexType(self, index_type: Optional[AlmostJson]) -> ir.IndexType:
        if index_type is None:
            raise ValueError("No Index Type Provided")

        values: dict = index_type.attributes
        lower: ast.Expression = self.visit(values.get("lower_bound"))
        upper: ast.Expression = self.visit(values.get("upper_bound"))

        if (_stride := values.get("stride")) is not None:
            stride = self.visit(_stride)
        else:
            stride = self.visit_IntLiteral(
                AlmostJson(
                    cls_name=ast.IntLiteral.get_key_name(),
                    attributes=dict(
                        span=AlmostJson(
                            cls_name=Span.__name__,
                            attributes=dict(
                                start_column=0,
                                end_column=0,
                                start_line=0,
                                end_line=0,
                                source=AlmostJson(
                                    cls_name=Source.__name__,
                                    attributes=dict(namespace="_null"),
                                ),
                            ),
                        ),
                        value=1,
                    ),
                )
            )

        return ir.IndexType(lower_bound=lower, upper_bound=upper, stride=stride)

    def visit_TupleType(self, tuple_type: Optional[AlmostJson]) -> ir.TupleType:
        if tuple_type is None:
            raise ValueError("No Tuple Type Provided.")

        values: dict = tuple_type.attributes
        if (v := values.get("types")) is None:
            raise ValueError("Invalid Tuple Type. No Type definitions of Elements.")

        types: List[ir.Type] = self.visit_sequence(v)

        return ir.TupleType(types=types)

    def visit_Identifier(self, identifier: Optional[AlmostJson]) -> ir.Identifier:
        if identifier is None:
            raise ValueError("No Identifier Provided")
        if (hint := identifier.attributes.get("name_hint")) is None:
            raise ValueError("Invalid Identifier Name")

        identity = ir.Identifier(name_hint=hint)

        if (value := identifier.attributes.get("_id")) is None:
            raise ValueError("Invalid ID.")
        identity._id = value

        # NOTE: We are Hacking the Identifier Class, which automatically assigns an ID.
        #       Do we care if we are increasing the assignment ID when creating these
        #       Objects? or Do we override the _id from the JSON?

        return identity

    def visit_sequence(self, nodes: Optional[List[AlmostJson]]) -> List[ASTObject]:
        if nodes is None:
            return []

        return [self.visit(node) for node in nodes]

    def visit_Span(self, span: Optional[AlmostJson]) -> Span:
        if span is None:
            return Span(0, 0, 0, 0)

        values: dict = span.attributes

        source: Optional[Source] = None
        if (_source := values.get("source")) is not None:
            source = self.visit_Source(_source)
        maybe = dict(source=source)
        return Span(
            start_column=values.get("start_column") or 0,
            end_column=values.get("end_column") or 0,
            start_line=values.get("start_line") or 0,
            end_line=values.get("end_line") or 0,
            **{k: v for k, v in maybe.items() if v is not None},
        )

    def visit_Source(self, source: AlmostJson) -> Source:
        return Source(namespace=source.attributes.get("namespace") or "_null")


def dump(node: ast.ASTNode, indent: Optional[str] = "  ") -> str:
    """Serialize an AST Node to json string, with a given indent."""
    to_json = ASTtoJSON()
    obj: AlmostJson = to_json.visit(node)

    return json.dumps(obj.data(), indent=indent)


def to_almost_json(obj: Any):
    """Json object hook to convert to AlmostJson object if matches criteria."""
    if isinstance(obj, dict) and set(obj.keys()) == {"cls_name", "attributes"}:
        return AlmostJson(
            cls_name=obj["cls_name"],
            attributes=obj["attributes"],
        )

    return obj


def load(json_string: str, **kwargs) -> Union[ast.ASTNode, List[ast.ASTNode]]:
    """Loads a Json string to construct an ASTNode."""
    node_almost: AlmostJson = json.loads(
        json_string, object_hook=to_almost_json, **kwargs
    )

    return JSONtoAST().visit(node_almost)
