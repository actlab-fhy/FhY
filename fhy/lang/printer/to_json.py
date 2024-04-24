"""Conversion of AST to json"""

import json
from typing import Any, Callable, List, Optional, Sequence

from fhy import ir
from fhy.lang import ast
from fhy.lang.ast import visitor
from fhy.lang.span import Source, Span


def get_cls_name(obj: Any) -> str:
    """Retrieves the Class name of an object."""
    if not hasattr(obj, "keyname"):
        return obj.__class__.__name__

    return obj.keyname()


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


class ASTtoJSON(visitor.Visitor):
    """Converts an AST Node into a json object"""

    def visit(self, node: visitor.ASTObject) -> Any:
        """A unified entry point that determines how to visit an AST object node"""
        name = f"visit_{get_cls_name(node)}"
        method: Callable[[visitor.ASTObject], Any] = getattr(self, name, self.default)

        return method(node)

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

        else:
            return super().default(node)

        return AlmostJson(cls_name=get_cls_name(node), attributes=attributes)

    def visit_Module(self, node: ast.Module) -> dict:
        components: List[dict] = []
        for comp in node.components:
            components.append(self.visit(comp))

        obj = AlmostJson(
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
            attributes=dict(
                span=self.visit_Span(node.span),
                expression=express,
            ),
        )

        return obj

    def visit_UnaryExpression(self, node: ast.UnaryExpression) -> AlmostJson:
        express = self.visit(node.expression)
        obj = AlmostJson(
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(node),
            attributes=dict(span=self.visit_Span(node.span), identifier=identifier),
        )

        return obj

    def visit_IntLiteral(self, node: ast.IntLiteral) -> AlmostJson:
        return AlmostJson(
            cls_name=get_cls_name(node),
            attributes=dict(span=self.visit_Span(node.span), value=node.value),
        )

    def visit_FloatLiteral(self, node: ast.FloatLiteral) -> AlmostJson:
        return AlmostJson(
            cls_name=get_cls_name(node),
            attributes=dict(span=self.visit_Span(node.span), value=node.value),
        )

    def visit_DataType(self, node: ir.DataType) -> AlmostJson:
        node.primitive_data_type.value

        return AlmostJson(
            cls_name=get_cls_name(node),
            attributes=dict(primitive_data_type=node.primitive_data_type.value),
        )

    def visit_QualifiedType(self, node: ast.QualifiedType) -> AlmostJson:
        base = self.visit(node.base_type)
        return AlmostJson(
            cls_name=get_cls_name(node),
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
            cls_name=get_cls_name(numerical_type),
            attributes=dict(data_type=dtype, shape=shape),
        )

    def visit_IndexType(self, index_type: ir.IndexType) -> AlmostJson:
        lower = self.visit(index_type.lower_bound)  # type: ignore[arg-type]
        upper = self.visit(index_type.upper_bound)  # type: ignore[arg-type]
        if index_type.stride is not None:
            stride = self.visit(index_type.stride)  # type: ignore[arg-type]
        else:
            stride = self.visit_IntLiteral(ast.IntLiteral(value=1))

        return AlmostJson(
            cls_name=get_cls_name(index_type),
            attributes=dict(lower_bound=lower, upper_bound=upper, stride=stride),
        )

    def visit_Identifier(self, identifier: ir.Identifier) -> AlmostJson:
        return AlmostJson(
            cls_name=get_cls_name(identifier),
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
            cls_name=get_cls_name(span),
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
            cls_name=get_cls_name(source), attributes=dict(namespace=source.namespace)
        )


def dump(node: ast.ASTNode, indent: str = "  ") -> str:
    """Serialize an AST Node to json string, with a given indent."""
    to_json = ASTtoJSON()
    obj = to_json.visit(node)
    return json.dumps(obj, indent=indent)


# TODO: Implement the Reverse process
def load(obj: dict, default: Callable):
    """Load a Json Object to construct an ASTNode"""
    ...
