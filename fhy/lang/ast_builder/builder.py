"""Deprecated Module.

A remnant of a decadent past with ambitious daliances in the joy and woeful misery of
using a stack + listener pattern to construct the AST from a CST parse tree.

NOTE: Module has been commented out, and will be removed.
TODO: Remove Module after relevant and desired code pieces have been scraped.

"""

# from copy import copy
# from dataclasses import replace
# from enum import StrEnum
# from typing import List, Optional, Type, Union

# from fhy import ir
# from fhy.ir import Type as IRType
# from fhy.lang.ast import (
#     Argument,
#     ASTNode,
#     Component,
#     Expression,
#     Function,
#     Module,
#     Operation,
#     Procedure,
#     QualifiedType,
#     Statement,
# )
# from fhy.lang.ast.expression import (
#     ArrayAccessExpression,
#     BinaryExpression,
#     BinaryOperation,
#     ComplexLiteral,
#     FloatLiteral,
#     FunctionExpression,
#     IdentifierExpression,
#     IntLiteral,
#     TernaryExpression,
#     TupleAccessExpression,
#     UnaryExpression,
#     UnaryOperation,
# )
# from fhy.lang.ast.statement import (
#     DeclarationStatement,
#     ExpressionStatement,
#     ForAllStatement,
#     ReturnStatement,
#     SelectionStatement,
# )
# from fhy.lang.span import Span
# from fhy.utils import Stack

# from .builder_frame import ASTBuilderFrame, create_builder_frame


# class ContextError(Exception):
#     """Unexpected Context Error"""

#     @classmethod
#     def message(
#         cls, context: str, node: str, obj: Union[ASTNode, Type]
#     ) -> "ContextError":
#         """Constructs General Context Closing Message"""
#         msg = f"Cannot close {context} context. "
#         msg += f"Expected {node} Node type, received: {obj}"
#         return cls(msg)


# class ASTBuilder(object):
#     """Controls the Stack of Nodes to construct a Proper AST Representation."""

# TODO: Static Typing of this Class is a Little Funny Right Now.
#       A Subclassed ASTNode violates Typing Assignment.
#       TypeVar["node", base_class=ASTNode] Doesn't work without explicitly
#       defining Functions to return correct subclasses.
# TODO Jason: Add docstring
#     _frame_stack: Stack[ASTBuilderFrame]
#     _ast: Optional[ASTNode]

#     def __init__(self):
#         self._frame_stack = Stack[ASTBuilderFrame]()
#         self._ast = None

#     @property
#     def ast(self) -> Optional[ASTNode]:
#         return self._ast

#     def set_current_frame_span(self, span: Span) -> None:
#         current_frame = self.get_current_frame()
#         if current_frame is None:
#             raise ContextError("Expected a builder frame to set span for")
#         current_frame.span = span

#     def get_current_frame(self) -> Optional[ASTBuilderFrame]:
#         if len(self._frame_stack) == 0:
#             return None
#         return self._frame_stack.peek()

#     def open_context(self, cls: type) -> None:
#         frame = create_builder_frame(cls)
#         self._frame_stack.push(frame)

#     def close_context(self) -> ASTBuilderFrame:
#         return self._frame_stack.pop()

# def close_module_context(self) -> None:
#     module_builder_frame: ASTBuilderFrame = self._frame_stack.pop()
#     if not issubclass(module_builder_frame.cls, Module):
#         raise ContextError.message(
#             "module", Module.keyname(), module_builder_frame
#         )
#     self._ast = module_builder_frame.build()

# def close_component_building(self) -> None:
#     component_builder_frame: ASTBuilderFrame = self._frame_stack.pop()
#     if not issubclass(component_builder_frame.cls, Component):
#         raise ContextError.message(
#             "component", Component.keyname(), component_builder_frame
#         )

#     module_builder_frame: ASTBuilderFrame = self.get_current_frame()
#     if not issubclass(module_builder_frame.cls, Module):
#         raise ContextError.message(
#             "component", Module.keyname(), module_builder_frame
#         )

#     module_builder_frame.components.append(component_builder_frame.build())

# def close_argument_building(self) -> None:
#     argument_builder_frame: ASTBuilderFrame = self._frame_stack.pop()
#     if not issubclass(argument_builder_frame.cls, Argument):
#         raise ContextError.message(
#             "argument", Argument.keyname(), argument_builder_frame
#         )

#     function_builder_frame: ASTBuilderFrame = self.get_current_frame()
#     if not issubclass(function_builder_frame.cls, Function):
#         raise ContextError.message(
#             "argument", Function.keyname(), function_builder_frame
#         )

#     function_builder_frame.args.append(argument_builder_frame.build())

# def close_qualified_type_building(self) -> None:
#     qualified_type_builder_frame: ASTBuilderFrame = self._frame_stack.pop()
#     if not issubclass(qualified_type_builder_frame.cls, QualifiedType):
#         raise ContextError.message(
#             "qualified type", QualifiedType.keyname(), qualified_type_builder_frame
#         )

#     parent_context_builder_frame: ASTBuilderFrame = self.get_current_frame()

#     qualified_type = qualified_type_builder_frame.build()
#     if issubclass(parent_context_builder_frame.cls, Operation):
#         parent_context_builder_frame.update(ret_type=qualified_type)

#     elif issubclass(parent_context_builder_frame.cls, Argument):
#         parent_context_builder_frame.update(qualified_type=qualified_type)

#     elif issubclass(parent_context_builder_frame.cls, Statement):
#         if issubclass(parent_context_builder_frame.cls, DeclarationStatement):
#             parent_context_builder_frame.update(_variable_type=qualified_type)
#         else:
#             raise NotImplementedError()

#     else:
#         raise ContextError.message(
#             "qualified type",
#             f"({Argument.keyname()} | {Operation.keyname()})",
#             parent_context_builder_frame,
#         )

# def add_dtype(self, dtype):
#     node: ASTNode = self.get_current_frame()

#     if not isinstance(node, Type):
#         raise ContextError(
#             f"Adding dtype. Current Node is not of type `Type`. Received: {node}"
#         )

#     data_type: PrimitiveDataType = validate(dtype, PrimitiveDataType)
#     node._data_type = DataType(data_type)

# def open_shape(self):
#     node: Type = self.get_current_frame()
#     if not isinstance(node, Type):
#         raise ContextError(
#             f"Opening Shape. Current Node is not of type `Type`. Received: {node}"
#         )
#     if not hasattr(node, "_shape") or not isinstance(node._shape, list):
#         node._shape = []

# def close_shape(self):
#     shape: List[ASTNode] = []
#     # NOTE: We don't know the number of elements defining shape
#     while len(self._frame_stack):
#         node: ASTNode = self._frame_stack.pop()
#         if isinstance(node, Type):
#             break
#         elif not isinstance(node, Expression):
#             raise ContextError.message("shape", Expression.keyname(), node)
#         shape.append(node)

#     if not isinstance(node, Type):
#         raise ContextError.message("shape", "`Type`", node)
#     node._shape = shape[::-1]
#     self._frame_stack.push(node)

# def close_type_building(self) -> None:
#     type_builder_frame: ASTBuilderFrame = self._frame_stack.pop()
#     if not issubclass(type_builder_frame.cls, ir.Type):
#         raise ContextError.message("type", "Type", type_builder_frame)

#     qualified_type_builder_frame: ASTBuilderFrame = self.get_current_frame()
#     if not isinstance(qualified_type_builder_frame.cls, QualifiedType):
#         raise ContextError.message(
#             "type", "QualifiedType", qualified_type_builder_frame
#         )

#     qualified_type_builder_frame.update(base_type=type_builder_frame.build())

# def open_declaration_statement(self, location: Span, name: str):
#     node = DeclarationStatement(
#         _span=location,
#         _variable_name=Identifier(name),
#         _variable_type=copy(MockQualifiedType),
#     )
#     self._frame_stack.push(node)

# def close_declaration_statement(self):
#     express: ASTNode = self._frame_stack.pop()

#     # Declaration Statment May Not Assign a Value.
#     if isinstance(express, DeclarationStatement):
#         self._frame_stack.push(express)
#         return

#     if not isinstance(express, Expression):
#         raise ContextError.message(
#             "declaration_statement", Expression.keyname(), express
#         )

#     current: ASTNode = self._frame_stack.pop()
#     if not isinstance(current, DeclarationStatement):
#         raise ContextError.message(
#             "declaration_statement", DeclarationStatement.keyname(), current
#         )

#     new_node = replace(current, _expression=express)
#     self._frame_stack.push(new_node)

# def open_expression_statement(self, location: Span, name: Optional[str]):
#     if name is None:
#         node = ExpressionStatement(_span=location, _left=None, _right=None)
#     else:
#         node = ExpressionStatement(
#             _span=location, _left=Identifier(name), _right=None
#         )
#     self._frame_stack.push(node)

# def close_expression_statement(self):
#     # TODO: This is not Collecting Correctly...
#     index = self._frame_stack.pop()
#     if not isinstance(index, ExpressionList):
#         raise ContextError.message(
#             "expression_statement", "ExpressionList", index
#         )

#     right = self._frame_stack.pop()
#     if not isinstance(right, Expression):
#         raise ContextError.message(
#             "expression_statement", Expression.keyname(), right
#         )

#     left = self._frame_stack.pop()
#     if not isinstance(left, Expression):
#         raise ContextError.message(
#             "expression_statement", Expression.keyname(), left
#         )

#     statement = self._frame_stack.pop()
#     if not isinstance(statement, ExpressionStatement):
#         raise ContextError.message(
#             "expression_statement", ExpressionStatement.keyname(), statement
#         )

#     new_node = replace(statement, _left=left, _right=right, _index=index.body)
#     self._frame_stack.push(new_node)

# def open_branch_statement(self):
#     node = BranchStatement(_predicate=Expression)
#     self._node_stack.push(node)

# def close_branch_statement(self):
#     # There are Variable Number of Expressions
#     # On Two Different Blocks that we need to
#     # Keep Track of Here... How do we do that?
#     # And blocks technically have no requirement
#     # to contain children...
#     ...

# def open_iteration_statement(self, location: Span):
#     node = ForAllStatement(_span=location, _index=Expression)
#     self._node_stack.push(node)

# def close_iteration_statement(self):
#     _index = self._node_stack.pop()
#     if not isinstance(_index, Expression):
#         raise ContextError.message(
#             "iteration_statement", Expression.keyname(), _index
#         )

#     _statement = self._node_stack.pop()
#     if not isinstance(_statement, ForAllStatement):
#         raise ContextError.message(
#             "iteration_statement", ForAllStatement.keyname(), _statement
#         )

#     new_node = replace(_statement, _index=_index)
#     self._node_stack.push(new_node)

# def open_return_statement(self, location: Span):
#     node = ReturnStatement(_span=location, _expression=Expression)
#     self._frame_stack.push(node)

# def close_return_statement(self):
#     express: ASTNode = self._frame_stack.pop()

#     # Return Statement May Not Have a Value.
#     if isinstance(express, ReturnStatement):
#         self._frame_stack.push(express)
#         return

#     if not isinstance(express, Expression):
#         raise ContextError.message(
#             "return_statement", Expression.keyname(), express
#         )

#     current: ASTNode = self._frame_stack.pop()
#     if not isinstance(current, ReturnStatement):
#         raise ContextError.message(
#             "return_statement", ReturnStatement.keyname(), current
#         )

#     new_node = replace(current, _expression=express)
#     self._frame_stack.push(new_node)

# def close_statement(self):
#     # NOTE: This is a General Close of Any Statement.
#     #       Each Statement Type Needs it's Own Prep to Close Correctly.
#     _statement: ASTNode = self._frame_stack.pop()
#     if not isinstance(_statement, Statement):
#         raise ContextError.message("statement", Statement.keyname(), _statement)

#     current: ASTNode = self._frame_stack.pop()
#     if not isinstance(current, (Function, Statement)):
#         raise ContextError.message(
#             "statement", f"{Function.keyname()} | {Statement.keyname()}", current
#         )

#     elif (
#         isinstance(current, Statement) and
#         not isinstance(current, ForAllStatement)
#         ):
#         raise NotImplementedError(
#             "Have Not Implemented CLosing Statements on anything other than"
#             f" ForAllStatements. Received: {current}"
#         )

#     body: List[Statement] = []
#     if hasattr(current, "body") and current.body is not None:
#         body.extend(current.body)
#     body.append(_statement)

#     new_node: Union[Function, Statement] = replace(current, body=body)
#     self._frame_stack.push(new_node)

# def open_expression_list(self):
#     self._frame_stack.push(ExpressionList())

# def close_expression_list(self):
#     body: List[Expression] = []
#     while len(self._frame_stack):
#         node = self._frame_stack.pop()
#         if isinstance(node, ExpressionList):
#             break
#         elif not isinstance(node, Expression):
#             raise ContextError.message(
#                 "expression_list", Expression.keyname(), node
#             )
#         body.append(node)

#     if not isinstance(node, ExpressionList):
#         raise ContextError.message(
#                 "expression_list", "ExpressionList", node
#             )

#     node.body = body[::-1]
#     self._frame_stack.push(node)

# def open_tensor_access_expression(self, location: Span):
#     node = ArrayAccessExpression(_span=location)
#     self._frame_stack.push(node)

# def close_tensor_access_expression(self):
#     express_body = self._frame_stack.pop()
#     if not isinstance(express_body, ExpressionList):
#         raise ContextError.message(
#             "tensor_access_expression", "ExpressionList", express_body
#         )

#     index_body = self._frame_stack.pop()
#     if not isinstance(index_body, ExpressionList):
#         raise ContextError.message(
#             "tensor_access_expression", "ExpressionList", index_body
#         )

#     tensor = self._frame_stack.pop()
#     if not isinstance(tensor, ArrayAccessExpression):
#         raise ContextError.message(
#             "tensor_access_expression", "ExpressionList", tensor
#         )

#     new_node = replace(
#         tensor,
#         _index=index_body.body,
#         _expressions=express_body.body,
#     )
#     self._frame_stack.push(new_node)

# def add_identifier(self, location: Span, name: str):
#     node = IdentifierExpression(_span=location, _identifier=Identifier(name))
#     self._frame_stack.push(node)

# def add_literal(self, location: Span, value: Union[int, float, complex]):
#     if isinstance(value, complex):
#         node = ComplexLiteral(_span=location, value=value)
#     elif isinstance(value, float):
#         node = FloatLiteral(_span=location, value=value)
#     elif isinstance(value, int):
#         node = IntLiteral(_span=location, value=value)
#     else:
#         raise NotImplementedError("Unknown Literal")
#     self._frame_stack.push(node)

# def add_unary_expression(self, location: Span, operator: str):
#     op = validate(operator, UnaryOperation)
#     node = UnaryExpression(_span=location, _operation=op, _expression=None)
#     self._frame_stack.push(node)

# def close_unary_expression(self):
#     expression_node: ASTNode = self._frame_stack.pop()
#     if not isinstance(expression_node, Expression):
#         raise ContextError.message(
#             "unary_expression", Expression.keyname(), expression_node
#         )

#     current: ASTNode = self._frame_stack.pop()
#     if not isinstance(current, UnaryExpression):
#         raise ContextError.message(
#             "unary_expression", UnaryExpression.keyname(), current
#         )

#     new_node = replace(current, _expression=expression_node)
#     self._frame_stack.push(new_node)

# def add_binary_expression(self, location: Span, operator: str):
#     op = validate(operator, BinaryOperation)
#     node = BinaryExpression(
#         _span=location, _operation=op, _left_expression=None, _right_expression=None
#     )

#     self._frame_stack.push(node)

# def close_binary_expression(self):
#     right: ASTNode = self._frame_stack.pop()
#     if not isinstance(right, Expression):
#         raise ContextError.message("binary_expression", Expression.keyname(), right)

#     left: ASTNode = self._frame_stack.pop()
#     if not isinstance(left, Expression):
#         raise ContextError.message("binary_expression", Expression.keyname(), left)

#     binary: ASTNode = self._frame_stack.pop()
#     if not isinstance(binary, BinaryExpression):
#         raise ContextError.message(
#             "binary_expression", BinaryExpression.keyname(), binary
#         )
#     new_node = replace(binary, _left_expression=left, _right_expression=right)
#     self._frame_stack.push(new_node)

# def open_ternary_expression(self, location: Span):
#     node = TernaryExpression(
#         _span=location,
#         _condition=Expression,
#         _true_expression=Expression,
#         _false_expression=Expression,
#     )
#     self._frame_stack.push(node)

# def close_ternary_expression(self):
#     _false = self._frame_stack.pop()
#     if not isinstance(_false, Expression):
#         raise ContextError.message(
#             "ternary_expression", f"false {Expression.keyname()}", _false
#         )

#     _true = self._frame_stack.pop()
#     if not isinstance(_true, Expression):
#         raise ContextError.message(
#             "ternary_expression", f"true {Expression.keyname()}", _true
#         )

#     _condition = self._frame_stack.pop()
#     if not isinstance(_condition, Expression):
#         raise ContextError.message(
#             "ternary_expression", f"condition {Expression.keyname()}", _condition
#         )

#     current = self._frame_stack.pop()
#     if not isinstance(current, TernaryExpression):
#         raise ContextError.message(
#             "ternary_expression", TernaryExpression.keyname(), current
#         )

#     new_node = replace(
#         current,
#         _condition=_condition,
#         _true_expression=_true,
#         _false_expression=_false,
#     )
#     self._frame_stack.push(new_node)
