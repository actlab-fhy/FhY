"""A Simple Visitor Pattern base class to visit ASTNode objects.

Classes:
    BasePass: abstract visitor pattern class
    Visitor:
    Listener:
    Transformer:

"""

from abc import ABC
from copy import copy
from typing import Any, Callable, List, Sequence, Union

from fhy import ir
from fhy.utils.alias import ASTObject

from ..span import Source, Span
from .core import Expression, Module, Statement
from .expression import (
    ArrayAccessExpression,
    BinaryExpression,
    FloatLiteral,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    TernaryExpression,
    TupleAccessExpression,
    TupleExpression,
    UnaryExpression,
)
from .qualified_type import QualifiedType
from .statement import (
    Argument,
    DeclarationStatement,
    ExpressionStatement,
    ForAllStatement,
    Import,
    Operation,
    Procedure,
    ReturnStatement,
    SelectionStatement,
)

# def iter_fields(node: ASTNode) -> Generator[Tuple[str, Any], None, None]:
#     """Iterates through Relevant Attributes of a Node.

#     Returns:
#         Tuple[str, Any]

#     """
#     for field in node.visit_attrs():
#         if not hasattr(node, field):
#             continue
#         yield field, getattr(node, field)


# def iter_children(node: ASTNode) -> Generator[ASTNode, None, None]:
#     """Yields all Direct Child Nodes"""
#     for _, field in iter_fields(node):
#         if isinstance(field, ASTNode):
#             yield field
#         elif isinstance(field, list):
#             for child in field:
#                 if isinstance(child, ASTNode):
#                     yield child


def get_cls_name(obj: Any) -> str:
    """Retrieves the Class name of an object instance."""
    if not hasattr(obj, "keyname"):
        return obj.__class__.__name__

    return obj.keyname()


class BasePass(ABC):
    """Abstract Visitor Pattern Class for Node objects.

    Args:
        is_recursive (bool): If true, recursively visit child nodes.

    """

    _is_recursive: bool

    def __init__(self, is_recursive: bool = True) -> None:
        self._is_recursive = is_recursive

    def __call__(self, node: Any, *args: Any, **kwargs: Any) -> Any:
        return self.visit(node)

    def visit(self, node: Any) -> Any:
        """A unified entry point that determines how to visit an AST object node."""
        name = f"visit_{get_cls_name(node)}"
        method: Callable[[Any], Any] = getattr(self, name, self.default)

        return method(node)

    def default(self, node: Any) -> Any:
        """Default node visiting method."""
        raise NotImplementedError(f"Node `{type(node)}` is not supported.")


# TODO: we should be using all "visit" for each child node as everything returns None
class Visitor(BasePass):
    """ASTObject Visitor Pattern Class."""

    def visit(self, node: Union[ASTObject, Sequence[ASTObject]]) -> None:
        if isinstance(node, list):
            self.visit_sequence(node)
        else:
            super().visit(node)

    def visit_Module(self, node: Module) -> None:
        self.visit(node.name)
        self.visit_sequence(node.statements)

    def visit_Import(self, node: Import) -> None:
        self.visit(node.name)

    def visit_Operation(self, node: Operation) -> None:
        self.visit_Identifier(node.name)
        self.visit_sequence(node.args)
        self.visit(node.return_type)
        self.visit_sequence(node.body)

    def visit_Procedure(self, node: Procedure) -> None:
        self.visit_Identifier(node.name)
        self.visit_sequence(node.args)
        self.visit_sequence(node.body)

    def visit_Argument(self, node: Argument) -> None:
        self.visit(node.qualified_type)
        if node.name is not None:
            self.visit(node.name)

    def visit_DeclarationStatement(self, node: DeclarationStatement) -> None:
        self.visit(node.variable_name)
        self.visit(node.variable_type)
        if node.expression is not None:
            self.visit(node.expression)

    def visit_ExpressionStatement(self, node: ExpressionStatement) -> None:
        if node.left is not None:
            self.visit(node.left)
        self.visit(node.right)

    def visit_SelectionStatement(self, node: SelectionStatement) -> None:
        self.visit(node.condition)
        self.visit_sequence(node.true_body)
        self.visit_sequence(node.false_body)

    def visit_ForAllStatement(self, node: ForAllStatement) -> None:
        self.visit(node.index)
        self.visit_sequence(node.body)

    def visit_ReturnStatement(self, node: ReturnStatement) -> None:
        self.visit(node.expression)

    def visit_UnaryExpression(self, node: UnaryExpression) -> None:
        self.visit(node.expression)

    def visit_BinaryExpression(self, node: BinaryExpression) -> None:
        self.visit(node.left)
        self.visit(node.right)

    def visit_TernaryExpression(self, node: TernaryExpression) -> None:
        self.visit(node.condition)
        self.visit(node.true)
        self.visit(node.false)

    def visit_FunctionExpression(self, node: FunctionExpression) -> None:
        self.visit(node.function)
        self.visit_sequence(node.template_types)
        self.visit_sequence(node.indices)
        self.visit_sequence(node.args)

    def visit_ArrayAccessExpression(self, node: ArrayAccessExpression) -> None:
        self.visit(node.array_expression)
        self.visit_sequence(node.indices)

    def visit_TupleExpression(self, node: TupleExpression) -> None:
        self.visit_sequence(node.expressions)

    def visit_TupleAccessExpression(self, node: TupleAccessExpression) -> None:
        self.visit(node.tuple_expression)
        self.visit_IntLiteral(node.element_index)

    def visit_IdentifierExpression(self, node: IdentifierExpression) -> None:
        self.visit(node.identifier)

    def visit_IntLiteral(self, node: IntLiteral) -> None: ...

    def visit_FloatLiteral(self, node: FloatLiteral) -> None: ...

    def visit_QualifiedType(self, node: QualifiedType) -> None:
        self.visit(node.base_type)
        self.visit(node.type_qualifier)

    def visit_NumericalType(self, numerical_type: ir.NumericalType) -> None:
        self.visit(numerical_type.data_type)
        self.visit_sequence(numerical_type.shape)

    def visit_IndexType(self, index_type: ir.IndexType) -> None:
        self.visit(index_type.lower_bound)
        self.visit(index_type.upper_bound)
        if index_type.stride is not None:
            self.visit(index_type.stride)

    def visit_DataType(self, data_type: ir.DataType) -> None:
        self.visit(data_type.primitive_data_type)

    def visit_TypeQualifier(self, type_qualifier: ir.TypeQualifier) -> None: ...

    def visit_PrimitiveDataType(self, primitive: ir.PrimitiveDataType) -> None: ...

    def visit_Identifier(self, identifier: ir.Identifier) -> None: ...

    def visit_sequence(self, nodes: Sequence[ASTObject]) -> None:
        for node in nodes:
            self.visit(node)

    def visit_Span(self, span: Span) -> None:
        self.visit_Source(span.source)

    def visit_Source(self, source: Source) -> None: ...


# TODO: this doesn't visit anything after the first node...
class Listener(BasePass):
    """ASTObject Listener Pattern Class."""

    def default(self, node: Union[ASTObject, Sequence[ASTObject]]) -> None:
        if isinstance(node, list):
            return self.enter_sequence(node)
        super().default(node)

    def enter_Module(self, node: Module) -> None: ...
    def exit_Module(self, node: Module) -> None: ...

    def enter_Operation(self, node: Operation) -> None: ...
    def exit_Operation(self, node: Operation) -> None: ...

    def enter_Procedure(self, node: Procedure) -> None: ...
    def exit_Procedure(self, node: Procedure) -> None: ...

    def enter_Argument(self, node: Argument) -> None: ...
    def exit_Argument(self, node: Argument) -> None: ...

    def enter_DeclarationStatement(self, node: DeclarationStatement) -> None: ...
    def exit_DeclarationStatement(self, node: DeclarationStatement) -> None: ...

    def enter_ExpressionStatement(self, node: ExpressionStatement) -> None: ...
    def exit_ExpressionStatement(self, node: ExpressionStatement) -> None: ...

    def enter_SelectionStatement(self, node: SelectionStatement) -> None: ...
    def exit_SelectionStatement(self, node: SelectionStatement) -> None: ...

    def enter_ForAllStatement(self, node: ForAllStatement) -> None: ...
    def exit_ForAllStatement(self, node: ForAllStatement) -> None: ...

    def enter_ReturnStatement(self, node: ReturnStatement) -> None: ...
    def exit_ReturnStatement(self, node: ReturnStatement) -> None: ...

    def enter_UnaryExpression(self, node: UnaryExpression) -> None: ...
    def exit_UnaryExpression(self, node: UnaryExpression) -> None: ...

    def enter_BinaryExpression(self, node: BinaryExpression) -> None: ...
    def exit_BinaryExpression(self, node: BinaryExpression) -> None: ...

    def enter_TernaryExpression(self, node: TernaryExpression) -> None: ...
    def exit_TernaryExpression(self, node: TernaryExpression) -> None: ...

    def enter_FunctionExpression(self, node: FunctionExpression) -> None: ...
    def exit_FunctionExpression(self, node: FunctionExpression) -> None: ...

    def enter_ArrayAccessExpression(self, node: ArrayAccessExpression) -> None: ...
    def exit_ArrayAccessExpression(self, node: ArrayAccessExpression) -> None: ...

    def enter_TupleExpression(self, node: TupleExpression) -> None: ...
    def exit_TupleExpression(self, node: TupleExpression) -> None: ...

    def enter_TupleAccessExpression(self, node: TupleAccessExpression) -> None: ...
    def exit_TupleAccessExpression(self, node: TupleAccessExpression) -> None: ...

    def enter_IdentifierExpression(self, node: IdentifierExpression) -> None: ...
    def exit_IdentifierExpression(self, node: IdentifierExpression) -> None: ...

    def enter_IntLiteral(self, node: IntLiteral) -> None: ...
    def exit_IntLiteral(self, node: IntLiteral) -> None: ...

    def enter_FloatLiteral(self, node: FloatLiteral) -> None: ...
    def exit_FloatLiteral(self, node: FloatLiteral) -> None: ...

    def enter_QualifiedType(self, node: QualifiedType) -> None: ...
    def exit_QualifiedType(self, node: QualifiedType) -> None: ...

    def enter_DataType(self, node: ir.DataType) -> None: ...
    def exit_DataType(self, node: ir.DataType) -> None: ...

    def enter_NumericalType(self, numerical_type: ir.NumericalType) -> None: ...
    def exit_NumericalType(self, numerical_type: ir.NumericalType) -> None: ...

    def enter_IndexType(self, index_type: ir.IndexType) -> None: ...
    def exit_IndexType(self, index_type: ir.IndexType) -> None: ...

    def enter_Identifier(self, identifier: ir.Identifier) -> None: ...
    def exit_Identifier(self, identifier: ir.Identifier) -> None: ...

    def enter_sequence(self, nodes: Sequence[ASTObject]) -> None: ...
    def exit_sequence(self, nodes: Sequence[ASTObject]) -> None: ...

    def enter_Span(self, span: Span) -> None: ...
    def exit_Span(self, span: Span) -> None: ...

    def enter_Source(self, source: Source) -> None: ...
    def exit_Source(self, source: Source) -> None: ...


class Transformer(BasePass):
    """ASTObject Transformer Pass Pattern."""

    def visit_Module(self, node: Module) -> Module:
        new_name = self.visit_Identifier(node.name)
        new_statements = self.visit_Statements(node.statements)

        return Module(name=new_name, statements=new_statements)

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
            raise NotImplementedError(f"Node `{type(node)}` is not supported.")

    def visit_Import(self, node: Import) -> Import:
        new_name = self.visit_Identifier(node.name)

        return Import(name=new_name)

    def visit_Operation(self, node: Operation) -> Operation:
        new_name = self.visit_Identifier(node.name)
        new_args = self.visit_Arguments(node.args)
        new_return_type = self.visit_QualifiedType(node.return_type)
        new_body: List[Statement] = self.visit_Statements(node.body)

        return Operation(
            name=new_name, args=new_args, return_type=new_return_type, body=new_body
        )

    def visit_Procedure(self, node: Procedure) -> Procedure:
        new_name = self.visit_Identifier(node.name)
        new_args = self.visit_Arguments(node.args)
        new_body = self.visit_Statements(node.body)

        return Procedure(name=new_name, args=new_args, body=new_body)

    def visit_Arguments(self, nodes: List[Argument]) -> List[Argument]:
        return [self.visit_Argument(node) for node in nodes]

    def visit_Argument(self, node: Argument) -> None:
        new_qualified_type = self.visit_QualifiedType(node.qualified_type)
        if node.name is not None:
            new_name = self.visit_Identifier(node.name)
        else:
            new_name = None

        return Argument(qualified_type=new_qualified_type, name=new_name)

    def visit_DeclarationStatement(
        self, node: DeclarationStatement
    ) -> DeclarationStatement:
        new_variable_name = self.visit_Identifier(node.variable_name)
        new_variable_type = self.visit_QualifiedType(node.variable_type)
        if node.expression is not None:
            new_expression = self.visit_Expression(node.expression)
        else:
            new_expression = None

        return DeclarationStatement(
            variable_name=new_variable_name,
            variable_type=new_variable_type,
            expression=new_expression,
        )

    def visit_ExpressionStatement(
        self, node: ExpressionStatement
    ) -> ExpressionStatement:
        if node.left is not None:
            new_left = self.visit_Expression(node.left)
        else:
            new_left = None
        new_right = self.visit(node.right)

        return ExpressionStatement(left=new_left, right=new_right)

    def visit_SelectionStatement(self, node: SelectionStatement) -> SelectionStatement:
        new_condition = self.visit_Expression(node.condition)
        new_true_body = self.visit_Statements(node.true_body)
        new_false_body = self.visit_Statements(node.false_body)

        return SelectionStatement(
            condition=new_condition, true_body=new_true_body, false_body=new_false_body
        )

    def visit_ForAllStatement(self, node: ForAllStatement) -> ForAllStatement:
        new_index = self.visit_Expression(node.index)
        new_body = self.visit_Statements(node.body)

        return ForAllStatement(index=new_index, body=new_body)

    def visit_ReturnStatement(self, node: ReturnStatement) -> ReturnStatement:
        new_expression = self.visit(node.expression)

        return ReturnStatement(expression=new_expression)

    def visit_Expressions(self, nodes: List[Expression]) -> List[Expression]:
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
        new_expression = self.visit_Expression(node.expression)

        return UnaryExpression(operation=node.operation, expression=new_expression)

    def visit_BinaryExpression(self, node: BinaryExpression) -> BinaryExpression:
        new_left = self.visit_Expression(node.left)
        new_right = self.visit_Expression(node.right)

        return BinaryExpression(
            operation=node.operation, left=new_left, right=new_right
        )

    def visit_TernaryExpression(self, node: TernaryExpression) -> TernaryExpression:
        new_condition = self.visit_Expression(node.condition)
        new_true = self.visit_Expression(node.true)
        new_false = self.visit_Expression(node.false)

        return TernaryExpression(
            condition=new_condition, true=new_true, false=new_false
        )

    def visit_FunctionExpression(self, node: FunctionExpression) -> FunctionExpression:
        new_function = self.visit_Expression(node.function)
        new_template_types = self.visit_Types(node.template_types)
        new_indices = self.visit_Expressions(node.indices)
        new_args = self.visit_Expressions(node.args)

        return FunctionExpression(
            function=new_function,
            template_types=new_template_types,
            indices=new_indices,
            args=new_args,
        )

    def visit_ArrayAccessExpression(
        self, node: ArrayAccessExpression
    ) -> ArrayAccessExpression:
        new_array_expresssion = self.visit_Expression(node.array_expression)
        new_indices = self.visit_Expressions(node.indices)

        return ArrayAccessExpression(
            array_expression=new_array_expresssion, indices=new_indices
        )

    def visit_TupleExpression(self, node: TupleExpression) -> TupleExpression:
        new_expressions = self.visit_Expressions(node.expressions)

        return TupleExpression(expressions=new_expressions)

    def visit_TupleAccessExpression(
        self, node: TupleAccessExpression
    ) -> TupleAccessExpression:
        new_tuple_expression = self.visit(node.tuple_expression)
        new_element_index = self.visit_IntLiteral(node.element_index)

        return TupleAccessExpression(
            tuple_expression=new_tuple_expression, element_index=new_element_index
        )

    def visit_IdentifierExpression(
        self, node: IdentifierExpression
    ) -> IdentifierExpression:
        new_identifier = self.visit_Identifier(node.identifier)
        return IdentifierExpression(identifier=new_identifier)

    def visit_IntLiteral(self, node: IntLiteral) -> IntLiteral:
        return copy(node)

    def visit_FloatLiteral(self, node: FloatLiteral) -> FloatLiteral:
        return copy(node)

    def visit_QualifiedType(self, node: QualifiedType) -> QualifiedType:
        new_base_type = self.visit_Type(node.base_type)
        new_type_qualifier = self.visit_TypeQualifier(node.type_qualifier)

        return QualifiedType(base_type=new_base_type, type_qualifier=new_type_qualifier)

    def visit_Types(self, nodes: List[ir.Type]) -> List[ir.Type]:
        return [self.visit_Type(node) for node in nodes]

    def visit_Type(self, node: ir.Type) -> ir.Type:
        if isinstance(node, ir.NumericalType):
            return self.visit_NumericalType(node)
        elif isinstance(node, ir.IndexType):
            return self.visit_IndexType(node)
        else:
            raise NotImplementedError(f'Node "{type(node)}" is not supported.')

    def visit_NumericalType(self, numerical_type: ir.NumericalType) -> ir.NumericalType:
        new_data_type = self.visit_DataType(numerical_type.data_type)
        new_shape = self.visit_Expressions(numerical_type.shape)

        return ir.NumericalType(data_type=new_data_type, shape=new_shape)

    def visit_IndexType(self, index_type: ir.IndexType) -> None:
        new_lower_bound = self.visit_Expression(index_type.lower_bound)
        new_upper_bound = self.visit_Expression(index_type.upper_bound)
        if index_type.stride is not None:
            new_stride = self.visit_Expression(index_type.stride)
        else:
            new_stride = None
        return ir.IndexType(
            lower_bound=new_lower_bound, upper_bound=new_upper_bound, stride=new_stride
        )

    def visit_DataType(self, data_type: ir.DataType) -> ir.DataType:
        new_primitive_data_type = self.visit(data_type.primitive_data_type)

        return ir.DataType(primitive_data_type=new_primitive_data_type)

    def visit_TypeQualifier(self, type_qualifier: ir.TypeQualifier) -> ir.TypeQualifier:
        return copy(type_qualifier)

    def visit_PrimitiveDataType(
        self, primitive: ir.PrimitiveDataType
    ) -> ir.PrimitiveDataType:
        return copy(primitive)

    def visit_Identifier(self, identifier: ir.Identifier) -> ir.Identifier:
        return copy(identifier)

    def visit_Span(self, span: Span) -> Span:
        # TODO: fix to copy everything
        # new_source = self.visit_Source(span.source)
        # return Span(source=new_source)
        return span

    def visit_Source(self, source: Source) -> Source:
        return copy(source)
