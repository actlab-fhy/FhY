"""Conversion of AST to json"""

import json
from typing import Any, Callable, List, Optional, Sequence, Union

from fhy import ir
from fhy.lang import ast
from fhy.lang.ast import visitor
from fhy.lang.span import Source, Span


def convert(value: object) -> Any:
    """Recursively Converts objects into dictionary records."""
    if isinstance(value, AlmostJson):
        return value.data()

    elif isinstance(value, list):
        return [convert(child) for child in value]

    else:
        return value


class AlmostJson:
    """Consistent Data Structure Format for JSON Preparations"""

    cls_name: str
    attributes: dict

    def __init__(self, cls_name: str, attributes: dict) -> None:
        self.cls_name = cls_name
        self.attributes = attributes

    def data(self) -> dict:
        """Returns class attributes into a dictionary record, recursively."""
        atrib = {k: convert(v) for k, v in self.attributes.items()}
        return dict(cls_name=self.cls_name, attributes=atrib)

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, AlmostJson)
            and self.cls_name == value.cls_name
            and self.attributes == value.attributes
        )


JSONObject = Union[AlmostJson, List[AlmostJson]]


class ASTtoJSON(visitor.BasePass):
    """Converts an AST Node into a json object"""

    def default(self, node: visitor.ASTObject) -> Any:
        if isinstance(node, list):
            return self.visit_sequence(node)

        attributes = {}
        if isinstance(node, ast.ASTNode):
            for field in node.visit_attrs():
                _value = getattr(node, field)
                if isinstance(_value, (visitor.ASTObject, list)):  # type: ignore[arg-type]
                    attributes[field] = self.visit(_value)
                else:
                    attributes[field] = _value

        elif isinstance(node, visitor.ASTObject):  # type: ignore[arg-type]
            for key, val in node.__dict__.items():
                if isinstance(val, (visitor.ASTObject, list)):  # type: ignore[arg-type]
                    attributes[key] = self.visit(val)
                else:
                    attributes[key] = val

        elif node is None:
            return node

        else:
            return super().default(node)

        return AlmostJson(cls_name=visitor.get_cls_name(node), attributes=attributes)

    def visit_Module(self, node: ast.Module) -> dict:
        components: List[AlmostJson] = self.visit_sequence(node.components)

        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                components=components,
            ),
        )

        return obj.data()

    def visit_Operation(self, node: ast.Operation) -> AlmostJson:
        args: List[AlmostJson] = self.visit_sequence(node.args)
        ret_type: AlmostJson = self.visit(node.return_type)
        body: List[AlmostJson] = self.visit_sequence(node.body)
        name = self.visit_Identifier(node.name)
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                name=name,
                args=args,
                return_type=ret_type,
                body=body,
            ),
        )

        return obj

    def visit_Procedure(self, node: ast.Procedure) -> AlmostJson:
        args: List[AlmostJson] = self.visit_sequence(node.args)
        body: List[AlmostJson] = self.visit_sequence(node.body)
        name: AlmostJson = self.visit_Identifier(node.name)

        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
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
        varname = self.visit(node.variable_name)
        vartype = self.visit(node.variable_type)
        express = self.visit(node.expression) if node.expression is not None else None
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
        left = self.visit(node.left) if node.left is not None else None
        right = self.visit(node.right)
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
        condition = self.visit(node.condition)
        tbody = self.visit_sequence(node.true_body)
        fbody = self.visit_sequence(node.false_body)
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
        index = self.visit(node.index)
        body = self.visit_sequence(node.body)
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
        express = self.visit(node.expression)
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                expression=express,
            ),
        )

        return obj

    def visit_UnaryExpression(self, node: ast.UnaryExpression) -> AlmostJson:
        express = self.visit(node.expression)
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
        left = self.visit(node.left)
        right = self.visit(node.right)

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
        condition = self.visit(node.condition)
        true = self.visit(node.true)
        false = self.visit(node.false)

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
        function = self.visit(node.function)
        template = self.visit_sequence(node.template_types)
        index = self.visit_sequence(node.indices)
        args = self.visit_sequence(node.args)
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
        array = self.visit(node.array_expression)
        index = self.visit_sequence(node.indices)
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
        express = self.visit_sequence(node.expressions)
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
        tple: Any = self.visit(node.tuple_expression)
        element = self.visit_IntLiteral(node.element_index)
        obj = AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                tuple_expression=tple,
                element_index=element,
            ),
        )

        return obj

    def visit_IdentifierExpression(self, node: ast.IdentifierExpression) -> AlmostJson:
        identifier = self.visit(node.identifier)

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

    def visit_DataType(self, node: ir.DataType) -> AlmostJson:
        node.primitive_data_type.value

        return AlmostJson(
            cls_name=visitor.get_cls_name(node),
            attributes=dict(primitive_data_type=node.primitive_data_type.value),
        )

    def visit_QualifiedType(self, node: ast.QualifiedType) -> AlmostJson:
        base = self.visit(node.base_type)

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
        shape: List[AlmostJson] = self.visit_sequence(numerical_type.shape)  # type: ignore[arg-type]

        return AlmostJson(
            cls_name=visitor.get_cls_name(numerical_type),
            attributes=dict(data_type=dtype, shape=shape),
        )

    def visit_IndexType(self, index_type: ir.IndexType) -> AlmostJson:
        lower = self.visit(index_type.lower_bound)  # type: ignore[arg-type]
        upper = self.visit(index_type.upper_bound)  # type: ignore[arg-type]
        if index_type.stride is not None:
            stride = self.visit(index_type.stride)  # type: ignore[arg-type]
        else:
            stride = self.visit_IntLiteral(
                ast.IntLiteral(value=1, span=Span(0, 0, 0, 0))
            )

        return AlmostJson(
            cls_name=visitor.get_cls_name(index_type),
            attributes=dict(lower_bound=lower, upper_bound=upper, stride=stride),
        )

    def visit_Identifier(self, identifier: ir.Identifier) -> AlmostJson:
        return AlmostJson(
            cls_name=visitor.get_cls_name(identifier),
            attributes=dict(name_hint=identifier.name_hint, _id=identifier.id),
        )

    def visit_sequence(self, nodes: Sequence[visitor.ASTObject]) -> List[AlmostJson]:
        return [self.visit(node) for node in nodes]

    def visit_Span(self, span: Span) -> AlmostJson:
        if span.source is not None:
            source = self.visit_Source(span.source)
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
    """Converts a JSON object into AST Nodes"""

    def visit(self, node: JSONObject):
        if isinstance(node, list):
            return self.visit_sequence(node)
        elif not isinstance(node, AlmostJson):
            return self.default(node)

        name = f"visit_{node.cls_name}"
        method: Callable[[JSONObject], visitor.ASTObject]
        method = getattr(self, name, self.default)

        return method(node)

    def default(self, node: Any) -> Any:
        return super().default(node)

    def visit_Module(self, node: AlmostJson) -> ast.Module:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        components: List[visitor.ASTObject]
        components = self.visit_sequence(values.get("components"))

        return ast.Module(span=span, components=components)

    def visit_Operation(self, node: AlmostJson) -> ast.Operation:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        args: List[visitor.ASTObject] = self.visit_sequence(values.get("args"))
        body: List[visitor.ASTObject] = self.visit_sequence(values.get("body"))
        name: ir.Identifier = self.visit_Identifier(values.get("name"))
        ret_type: ast.QualifiedType = self.visit_QualifiedType(
            values.get("return_type")
        )

        return ast.Operation(
            span=span, name=name, args=args, body=body, return_type=ret_type
        )

    def visit_Procedure(self, node: AlmostJson) -> ast.Procedure:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        args: List[visitor.ASTObject] = self.visit_sequence(values.get("args"))
        body: List[visitor.ASTObject] = self.visit_sequence(values.get("body"))
        name: ir.Identifier = self.visit_Identifier(values.get("name"))

        return ast.Procedure(span=span, name=name, args=args, body=body)

    def visit_Argument(self, node: AlmostJson) -> ast.Argument:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        qtype: ast.QualifiedType = self.visit_QualifiedType(
            values.get("qualified_type")
        )

        if (nombre := values.get("name")) is not None:
            values["name"] = self.visit_Identifier(nombre)
        name: Optional[ir.Identifier] = values.get("name")

        return ast.Argument(span=span, name=name, qualified_type=qtype)

    def visit_DeclarationStatement(self, node: AlmostJson) -> ast.DeclarationStatement:
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

    def visit_ExpressionStatement(self, node: AlmostJson) -> ast.ExpressionStatement:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        if (_left := values.get("left")) is not None:
            values["left"] = self.visit(_left)
        left: Optional[ast.Expression] = values.get("left")
        right: ast.Expression = self.visit(values.get("right"))

        return ast.ExpressionStatement(span=span, left=left, right=right)

    def visit_SelectionStatement(self, node: AlmostJson) -> ast.SelectionStatement:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        condition: ast.Expression = self.visit(values.get("condition"))
        tbody: List[visitor.ASTObject] = self.visit_sequence(values.get("true_body"))
        fbody: List[visitor.ASTObject] = self.visit_sequence(values.get("false_body"))

        return ast.SelectionStatement(
            span=span, condition=condition, true_body=tbody, false_body=fbody
        )

    def visit_ForAllStatement(self, node: AlmostJson) -> ast.ForAllStatement:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        index: ast.Expression = self.visit(values.get("index"))
        body: List[visitor.ASTObject] = self.visit_sequence(values.get("body"))

        return ast.ForAllStatement(span=span, index=index, body=body)

    def visit_ReturnStatement(self, node: AlmostJson) -> ast.ReturnStatement:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        express: ast.Expression = self.visit(values.get("expression"))

        return ast.ReturnStatement(span=span, expression=express)

    def visit_UnaryExpression(self, node: AlmostJson) -> ast.UnaryExpression:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        express: ast.Expression = self.visit(values.get("expression"))
        operator: ast.UnaryOperation = ast.UnaryOperation(values.get("operation"))

        return ast.UnaryExpression(span=span, operation=operator, expression=express)

    def visit_BinaryExpression(self, node: AlmostJson) -> ast.BinaryExpression:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        left: ast.Expression = self.visit(values.get("left"))
        right: ast.Expression = self.visit(values.get("right"))
        operator = ast.BinaryOperation(values.get("operation"))

        return ast.BinaryExpression(
            span=span, left=left, right=right, operation=operator
        )

    def visit_TernaryExpression(self, node: AlmostJson) -> ast.TernaryExpression:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        condition: ast.Expression = self.visit(values.get("condition"))
        true: ast.Expression = self.visit(values.get("true"))
        false: ast.Expression = self.visit(values.get("false"))

        return ast.TernaryExpression(
            span=span, condition=condition, true=true, false=false
        )

    def visit_FunctionExpression(self, node: AlmostJson) -> ast.FunctionExpression:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        function: ast.Expression = self.visit(values.get("function"))
        template: List[visitor.ASTObject] = self.visit_sequence(
            values.get("template_types")
        )
        index: List[visitor.ASTObject] = self.visit_sequence(values.get("indices"))
        args: List[visitor.ASTObject] = self.visit_sequence(values.get("args"))

        return ast.FunctionExpression(
            span=span,
            function=function,
            template_types=template,
            indices=index,
            args=args,
        )

    def visit_ArrayAccessExpression(
        self, node: AlmostJson
    ) -> ast.ArrayAccessExpression:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        array = self.visit(values.get("array_expression"))
        index = self.visit_sequence(values.get("indices"))

        return ast.ArrayAccessExpression(
            span=span,
            array_expression=array,
            indices=index,
        )

    def visit_TupleExpression(self, node: AlmostJson) -> ast.TupleExpression:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        express = self.visit_sequence(values.get("expressions"))

        return ast.TupleExpression(span=span, expressions=express)

    def visit_TupleAccessExpression(
        self, node: AlmostJson
    ) -> ast.TupleAccessExpression:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        texpress: Any = self.visit(values.get("tuple_expression"))
        element = self.visit_IntLiteral(values.get("element_index"))

        return ast.TupleAccessExpression(
            span=span, tuple_expression=texpress, element_index=element
        )

    def visit_IdentifierExpression(self, node: AlmostJson) -> ast.IdentifierExpression:
        values: dict = node.attributes
        identifier = self.visit_Identifier(values.get("identifier"))
        span: Span = self.visit_Span(values.get("span"))

        return ast.IdentifierExpression(span=span, identifier=identifier)

    def visit_IntLiteral(self, node: AlmostJson) -> ast.IntLiteral:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        value = values.get("value")

        return ast.IntLiteral(span=span, value=value)

    def visit_FloatLiteral(self, node: AlmostJson) -> ast.FloatLiteral:
        values: dict = node.attributes
        span: Span = self.visit_Span(values.get("span"))
        value = values.get("value")

        return ast.FloatLiteral(span=span, value=value)

    def visit_DataType(self, node: AlmostJson) -> ir.DataType:
        primitive = ir.PrimitiveDataType(node.attributes.get("primitive_data_type"))

        return ir.DataType(primitive_data_type=primitive)

    def visit_QualifiedType(self, node: AlmostJson) -> ast.QualifiedType:
        values: dict = node.attributes
        base = self.visit(values.get("base_type"))
        span: Span = self.visit_Span(values.get("span"))
        qtype = ir.TypeQualifier(values.get("type_qualifier"))

        return ast.QualifiedType(
            span=span,
            base_type=base,
            type_qualifier=qtype,
        )

    def visit_NumericalType(self, numerical_type: AlmostJson) -> ir.NumericalType:
        values: dict = numerical_type.attributes
        dtype: ir.DataType = self.visit_DataType(values.get("data_type"))
        shape: List[visitor.ASTObject] = self.visit_sequence(values.get("shape"))

        return ir.NumericalType(data_type=dtype, shape=shape)

    def visit_IndexType(self, index_type: AlmostJson) -> ir.IndexType:
        values: dict = index_type.attributes
        lower = self.visit(values.get("lower_bound"))
        upper = self.visit(values.get("upper_bound"))

        if (_stride := values.get("stride")) is not None:
            stride = self.visit(_stride)
        else:
            stride = self.visit_IntLiteral(
                ast.IntLiteral(value=1, span=Span(0, 0, 0, 0))
            )

        return ir.IndexType(lower_bound=lower, upper_bound=upper, stride=stride)

    def visit_Identifier(self, identifier: AlmostJson) -> ir.Identifier:
        _id = ir.Identifier(name_hint=identifier.attributes.get("name_hint"))
        _id._id = identifier.attributes.get("_id")

        # NOTE: We are Hacking the Identifier Class, which automatically assigns an ID.
        #       Do we care if we are increasing the assignment ID when creating these
        #       Objects? or Do we override the _id from the JSON?

        return _id

    def visit_sequence(self, nodes: List[AlmostJson]) -> List[visitor.ASTObject]:
        return [self.visit(node) for node in nodes]

    def visit_Span(self, span: AlmostJson) -> Span:
        values: dict = span.attributes

        if (_source := values.get("source")) is not None:
            source = self.visit_Source(_source)
            values["source"] = source

        return Span(
            start_column=values.get("start_column"),
            end_column=values.get("end_column"),
            start_line=values.get("start_line"),
            end_line=values.get("end_line"),
            source=source,
        )

    def visit_Source(self, source: AlmostJson) -> Source:
        return Source(namespace=source.attributes.get("namespace"))


def dump(node: ast.ASTNode, indent: str = "  ") -> str:
    """Serialize an AST Node to json string, with a given indent."""
    to_json = ASTtoJSON()
    obj = to_json.visit(node)
    if isinstance(obj, AlmostJson):
        return json.dumps(obj.data(), indent=indent)
    return json.dumps(obj, indent=indent)


def to_almost_json(obj):
    if "cls_name" in obj and "attributes" in obj:
        return AlmostJson(
            cls_name=obj["cls_name"],
            attributes=obj["attributes"],
        )
    return obj


def load(json_string: str, **kwargs) -> Union[ast.ASTNode, List[ast.ASTNode]]:
    """Loads a Json string to construct an ASTNode"""
    node_almost: AlmostJson = json.loads(
        json_string, object_hook=to_almost_json, **kwargs
    )
    return JSONtoAST().visit(node_almost)
