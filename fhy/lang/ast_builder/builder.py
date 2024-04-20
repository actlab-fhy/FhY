# TODO Jason: Add docstring
from copy import copy
from dataclasses import replace
from enum import StrEnum
from typing import List, Optional, Union

from fhy.ir import (
    DataType,
    Identifier,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
)
from fhy.lang.ast import (
    Argument,
    ASTNode,
    Component,
    Expression,
    Function,
    Module,
    Operation,
    Procedure,
    QualifiedType,
    Statement,
)
from fhy.lang.ast.expression import (
    BinaryExpression,
    BinaryOperation,
    ComplexLiteral,
    ExpressionList,
    FloatLiteral,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    TernaryExpression,
    TensorAccessExpression,
    TupleAccessExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy.lang.ast.statement import (
    BranchStatement,
    DeclarationStatement,
    ExpressionStatement,
    ForAllStatement,
    ReturnStatement,
)
from fhy.lang.span import Span
from fhy.utils import Stack


def validate(name: str, enumeration: StrEnum) -> StrEnum:
    """Retrieves the Value from a Defined String Enumeration, and performs simple checks
    to validate the value.

    Args:
        name (str): the key name of the enumeration
        enumeration (StrEnum): An Enumeration of Supported Values

    Returns:
        (StrEnum) An instance of the provided enumeration key name.

    Raises:
        ValueError: When the name is not found within the enumeration.

    """
    try:
        value = enumeration(name)
    except ValueError:
        raise ValueError(f"Unsupported {enumeration.__name__}: {name}")  # type: ignore[attr-defined]

    return value


class ContextError(Exception):
    """Unexpected Context Error"""

    @classmethod
    def message(
        cls, context: str, node: str, obj: Union[ASTNode, Type]
    ) -> "ContextError":
        """Constructs General Context Closing Message"""
        msg = f"Cannot close {context} context. "
        msg += f"Expected {node} Node type, received: {obj}"
        return cls(msg)


# Simple Mock Types, to Temporarily instantiate a Placeholder Type
# NOTE: This may not be the best Strategy Here. The corrolary, is
#       that it might not be a good Idea to make AST Node Fields
#       Optional, since that is only true during build, not after creation.
_MockType = PrimitiveDataType._PLACEHOLDER
_MockQual = TypeQualifier._PLACEHOLDER
MockQualifiedType = QualifiedType(
    _span=None, base_type=_MockType, type_qualifier=_MockQual
)


class ASTBuilder(object):
    """Controls the Stack of Nodes to construct a Proper AST Representation."""

    # TODO: Static Typing of this Class is a Little Funny Right Now.
    #       A Subclassed ASTNode violates Typing Assignment.
    #       TypeVar["node", base_class=ASTNode] Doesn't work without explicitly defining
    #       Functions to return correct subclasses.
    # TODO Jason: Add docstring
    _node_stack: Stack[ASTNode]
    _ast: Optional[ASTNode]

    def __init__(self):
        self._node_stack = Stack[ASTNode]()
        self._ast = None

    @property
    def ast(self) -> Optional[ASTNode]:
        return self._ast

    def get_current_node(self) -> Optional[ASTNode]:
        if len(self._node_stack) == 0:
            return None
        return self._node_stack.peek()

    def add_module(self) -> None:
        self._node_stack.push(Module(_span=None))

    def close_module_building(self) -> None:
        module_node: ASTNode = self._node_stack.pop()
        if not isinstance(module_node, Module):
            raise ContextError.message("module", Module.keyname(), module_node)

        self._ast = module_node

    def add_procedure(self, location: Span, name: str) -> None:
        node = Procedure(_span=location, name=Identifier(name))
        self._node_stack.push(node)

    def add_operation(self, location: Span, name: str) -> None:
        node = Operation(
            _span=location, name=Identifier(name), ret_type=copy(MockQualifiedType)
        )
        self._node_stack.push(node)

    def close_component_building(self) -> None:
        component_node: ASTNode = self._node_stack.pop()
        if not isinstance(component_node, Component):
            raise ContextError.message("component", Component.keyname(), component_node)

        module_node: ASTNode = self._node_stack.pop()
        if not isinstance(module_node, Module):
            raise ContextError.message("component", Module.keyname(), module_node)

        components: List[Component] = module_node.components
        components.append(component_node)
        new_module_node: Module = replace(module_node, components=components)
        self._node_stack.push(new_module_node)

    def add_argument(self, location: Span, arg_name: str) -> None:
        node = Argument(_span=location, name=Identifier(arg_name))
        self._node_stack.push(node)

    def close_argument_building(self) -> None:
        argument_node: ASTNode = self._node_stack.pop()
        if not isinstance(argument_node, Argument):
            raise ContextError.message("argument", Argument.keyname(), argument_node)

        function_node: ASTNode = self._node_stack.pop()
        if not isinstance(function_node, Function):
            raise ContextError.message("argument", Function.keyname(), function_node)

        args: List[Argument] = function_node.args[:]  # type: ignore[attr-defined]
        args.append(argument_node)
        new_function_node: Component = replace(function_node, args=args)  # type: ignore[call-arg]
        self._node_stack.push(new_function_node)

    def add_qualified_type(self, location: Span, name: str) -> None:
        qualified_type: TypeQualifier = validate(name, TypeQualifier)  # type: ignore
        node = QualifiedType(
            _span=location, base_type=_MockType, type_qualifier=qualified_type
        )

        # TODO: We are pushing a Type onto the stack, instead of an AST Node
        #       Can we / Should we Instead modify the previous node?
        self._node_stack.push(node)

    def close_qualified_type_building(self) -> None:
        qualified_type_node: ASTNode = self._node_stack.pop()
        if not isinstance(qualified_type_node, QualifiedType):
            raise ContextError.message(
                "qualified type", QualifiedType.keyname(), qualified_type_node
            )

        previous_node: ASTNode = self._node_stack.pop()

        # Support Return Types on Operations
        if isinstance(previous_node, Operation):
            kwargs = dict(ret_type=qualified_type_node)

        elif isinstance(previous_node, Argument):
            kwargs = dict(qualified_type=qualified_type_node)

        elif isinstance(previous_node, Statement):
            if isinstance(previous_node, DeclarationStatement):
                kwargs = dict(_variable_type=qualified_type_node)
            else:
                raise NotImplementedError(
                    f"Other Statements Not Implemented Yet: {previous_node}"
                )

        else:
            raise ContextError.message(
                "qualified type",
                f"({Argument.keyname()} | {Operation.keyname()})",
                previous_node,
            )

        new_node: ASTNode = replace(previous_node, **kwargs)  # type: ignore[arg-type]
        self._node_stack.push(new_node)

    def add_dtype(self, dtype):
        node: ASTNode = self.get_current_node()

        if not isinstance(node, Type):
            raise ContextError(
                f"Adding dtype. Current Node is not of type `Type`. Received: {node}"
            )

        data_type: PrimitiveDataType = validate(dtype, PrimitiveDataType)
        node._data_type = DataType(data_type)

    def open_numerical_type(self) -> None:
        self._node_stack.push(NumericalType(_MockType, []))

    def open_shape(self):
        node: Type = self.get_current_node()
        if not isinstance(node, Type):
            raise ContextError(
                f"Opening Shape. Current Node is not of type `Type`. Received: {node}"
            )
        if not hasattr(node, "_shape") or not isinstance(node._shape, list):
            node._shape = []

    def close_shape(self):
        shape: List[ASTNode] = []
        # NOTE: We don't know the number of elements defining shape
        while len(self._node_stack):
            node: ASTNode = self._node_stack.pop()
            if isinstance(node, Type):
                break
            elif not isinstance(node, Expression):
                raise ContextError.message("shape", Expression.keyname(), node)
            shape.append(node)

        if not isinstance(node, Type):
            raise ContextError.message("shape", "`Type`", node)
        node._shape = shape[::-1]
        self._node_stack.push(node)

    def open_index_type(self) -> None:
        self._node_stack.push(IndexType(Expression, Expression, None))

    def _close_index_type(
        self,
        index: IndexType,
        low: Expression,
        high: Expression,
        stride: Optional[Expression],
    ) -> None:
        index._lower_bound = low
        index._upper_bound = high
        index._stride = stride

        self._node_stack.push(index)

    def close_index_type(self) -> None:
        stride_or_upper: ASTNode = self._node_stack.pop()
        if not isinstance(stride_or_upper, Expression):
            raise ContextError.message(
                "index_type", Expression.keyname(), stride_or_upper
            )

        upper_or_lower: ASTNode = self._node_stack.pop()
        if not isinstance(upper_or_lower, Expression):
            raise ContextError.message(
                "index_type", Expression.keyname(), upper_or_lower
            )

        lower_or_index: ASTNode = self._node_stack.pop()
        if isinstance(lower_or_index, IndexType):
            self._close_index_type(
                lower_or_index, upper_or_lower, stride_or_upper, None
            )
            return

        elif not isinstance(lower_or_index, Expression):
            raise ContextError.message(
                "index_type", Expression.keyname(), lower_or_index
            )

        index: ASTNode = self._node_stack.pop()
        if not isinstance(index, IndexType):
            raise ContextError.message("index_type", IndexType.keyname(), index)
        self._close_index_type(index, lower_or_index, upper_or_lower, stride_or_upper)

    def close_type_building(self) -> None:
        type_node: Type = self._node_stack.pop()
        if not isinstance(type_node, Type):
            raise ContextError.message("type", "Type", type_node)

        qualified_type_node: QualifiedType = self._node_stack.pop()
        if not isinstance(qualified_type_node, QualifiedType):
            raise ContextError.message("type", "QualifiedType", qualified_type_node)

        new_node = replace(qualified_type_node, base_type=type_node)
        self._node_stack.push(new_node)

    def open_declaration_statement(self, location: Span, name: str):
        node = DeclarationStatement(
            _span=location,
            _variable_name=Identifier(name),
            _variable_type=copy(MockQualifiedType),
        )
        self._node_stack.push(node)

    def close_declaration_statement(self):
        express: ASTNode = self._node_stack.pop()

        # Declaration Statment May Not Assign a Value.
        if isinstance(express, DeclarationStatement):
            self._node_stack.push(express)
            return

        if not isinstance(express, Expression):
            raise ContextError.message(
                "declaration_statement", Expression.keyname(), express
            )

        current: ASTNode = self._node_stack.pop()
        if not isinstance(current, DeclarationStatement):
            raise ContextError.message(
                "declaration_statement", DeclarationStatement.keyname(), current
            )

        new_node = replace(current, _expression=express)
        self._node_stack.push(new_node)

    def open_expression_statement(self, location: Span, name: Optional[str]):
        if name is None:
            node = ExpressionStatement(_span=location, _left=None, _right=None)
        else:
            node = ExpressionStatement(
                _span=location, _left=Identifier(name), _right=None
            )
        self._node_stack.push(node)

    def close_expression_statement(self):
        # TODO: This is not Collecting Correctly...
        index = self._node_stack.pop()
        if not isinstance(index, ExpressionList):
            raise ContextError.message(
                "expression_statement", "ExpressionList", index
            )

        right = self._node_stack.pop()
        if not isinstance(right, Expression):
            raise ContextError.message(
                "expression_statement", Expression.keyname(), right
            )

        left = self._node_stack.pop()
        if not isinstance(left, Expression):
            raise ContextError.message(
                "expression_statement", Expression.keyname(), left
            )

        statement = self._node_stack.pop()
        if not isinstance(statement, ExpressionStatement):
            raise ContextError.message(
                "expression_statement", ExpressionStatement.keyname(), statement
            )

        new_node = replace(statement, _left=left, _right=right, _index=index.body)
        self._node_stack.push(new_node)

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

    def open_return_statement(self, location: Span):
        node = ReturnStatement(_span=location, _expression=Expression)
        self._node_stack.push(node)

    def close_return_statement(self):
        express: ASTNode = self._node_stack.pop()

        # Return Statement May Not Have a Value.
        if isinstance(express, ReturnStatement):
            self._node_stack.push(express)
            return

        if not isinstance(express, Expression):
            raise ContextError.message(
                "return_statement", Expression.keyname(), express
            )

        current: ASTNode = self._node_stack.pop()
        if not isinstance(current, ReturnStatement):
            raise ContextError.message(
                "return_statement", ReturnStatement.keyname(), current
            )

        new_node = replace(current, _expression=express)
        self._node_stack.push(new_node)

    def close_statement(self):
        # NOTE: This is a General Close of Any Statement.
        #       Each Statement Type Needs it's Own Prep to Close Correctly.
        _statement: ASTNode = self._node_stack.pop()
        if not isinstance(_statement, Statement):
            raise ContextError.message("statement", Statement.keyname(), _statement)

        current: ASTNode = self._node_stack.pop()
        if not isinstance(current, (Function, Statement)):
            raise ContextError.message(
                "statement", f"{Function.keyname()} | {Statement.keyname()}", current
            )

        elif (
            isinstance(current, Statement) and
            not isinstance(current, ForAllStatement)
            ):
            raise NotImplementedError(
                "Have Not Implemented CLosing Statements on anything other than"
                f" ForAllStatements. Received: {current}"
            )   

        body: List[Statement] = []
        if hasattr(current, "body") and current.body is not None:
            body.extend(current.body)
        body.append(_statement)

        new_node: Union[Function, Statement] = replace(current, body=body)
        self._node_stack.push(new_node)

    def open_expression_list(self):
        self._node_stack.push(ExpressionList())

    def close_expression_list(self):
        body: List[Expression] = []
        while len(self._node_stack):
            node = self._node_stack.pop()
            if isinstance(node, ExpressionList):
                break
            elif not isinstance(node, Expression):
                raise ContextError.message(
                    "expression_list", Expression.keyname(), node
                )
            body.append(node)

        if not isinstance(node, ExpressionList):
            raise ContextError.message(
                    "expression_list", "ExpressionList", node
                )

        node.body = body[::-1]
        self._node_stack.push(node)

    def open_tensor_access_expression(self, location: Span):
        node = TensorAccessExpression(_span=location)
        self._node_stack.push(node)

    def close_tensor_access_expression(self):
        express_body = self._node_stack.pop()
        if not isinstance(express_body, ExpressionList):
            raise ContextError.message(
                "tensor_access_expression", "ExpressionList", express_body
            )

        index_body = self._node_stack.pop()
        if not isinstance(index_body, ExpressionList):
            raise ContextError.message(
                "tensor_access_expression", "ExpressionList", index_body
            )

        tensor = self._node_stack.pop()
        if not isinstance(tensor, TensorAccessExpression):
            raise ContextError.message(
                "tensor_access_expression", "ExpressionList", tensor
            )

        new_node = replace(
            tensor,
            _index=index_body.body,
            _expressions=express_body.body,
        )
        self._node_stack.push(new_node)

    def add_identifier(self, location: Span, name: str):
        node = IdentifierExpression(_span=location, _identifier=Identifier(name))
        self._node_stack.push(node)

    def add_literal(self, location: Span, value: Union[int, float, complex]):
        if isinstance(value, complex):
            node = ComplexLiteral(_span=location, value=value)
        elif isinstance(value, float):
            node = FloatLiteral(_span=location, value=value)
        elif isinstance(value, int):
            node = IntLiteral(_span=location, value=value)
        else:
            raise NotImplementedError("Unknown Literal")
        self._node_stack.push(node)

    def add_unary_expression(self, location: Span, operator: str):
        op = validate(operator, UnaryOperation)
        node = UnaryExpression(_span=location, _operation=op, _expression=None)
        self._node_stack.push(node)

    def close_unary_expression(self):
        expression_node: ASTNode = self._node_stack.pop()
        if not isinstance(expression_node, Expression):
            raise ContextError.message(
                "unary_expression", Expression.keyname(), expression_node
            )

        current: ASTNode = self._node_stack.pop()
        if not isinstance(current, UnaryExpression):
            raise ContextError.message(
                "unary_expression", UnaryExpression.keyname(), current
            )

        new_node = replace(current, _expression=expression_node)
        self._node_stack.push(new_node)

    def add_binary_expression(self, location: Span, operator: str):
        op = validate(operator, BinaryOperation)
        node = BinaryExpression(
            _span=location, _operation=op, _left_expression=None, _right_expression=None
        )

        self._node_stack.push(node)

    def close_binary_expression(self):
        right: ASTNode = self._node_stack.pop()
        if not isinstance(right, Expression):
            raise ContextError.message("binary_expression", Expression.keyname(), right)

        left: ASTNode = self._node_stack.pop()
        if not isinstance(left, Expression):
            raise ContextError.message("binary_expression", Expression.keyname(), left)

        binary: ASTNode = self._node_stack.pop()
        if not isinstance(binary, BinaryExpression):
            raise ContextError.message(
                "binary_expression", BinaryExpression.keyname(), binary
            )
        new_node = replace(binary, _left_expression=left, _right_expression=right)
        self._node_stack.push(new_node)

    def open_ternary_expression(self, location: Span):
        node = TernaryExpression(
            _span=location,
            _condition=Expression,
            _true_expression=Expression,
            _false_expression=Expression,
        )
        self._node_stack.push(node)

    def close_ternary_expression(self):
        _false = self._node_stack.pop()
        if not isinstance(_false, Expression):
            raise ContextError.message(
                "ternary_expression", f"false {Expression.keyname()}", _false
            )

        _true = self._node_stack.pop()
        if not isinstance(_true, Expression):
            raise ContextError.message(
                "ternary_expression", f"true {Expression.keyname()}", _true
            )

        _condition = self._node_stack.pop()
        if not isinstance(_condition, Expression):
            raise ContextError.message(
                "ternary_expression", f"condition {Expression.keyname()}", _condition
            )

        current = self._node_stack.pop()
        if not isinstance(current, TernaryExpression):
            raise ContextError.message(
                "ternary_expression", TernaryExpression.keyname(), current
            )

        new_node = replace(
            current,
            _condition=_condition,
            _true_expression=_true,
            _false_expression=_false,
        )
        self._node_stack.push(new_node)
