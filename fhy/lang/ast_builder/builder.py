# TODO Jason: Add docstring
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
    FloatLiteral,
    IdentifierExpression,
    IntLiteral,
    TernaryExpression,
    UnaryExpression,
    UnaryOperation
)
from fhy.lang.ast.statement import (
    BranchStatement,
    DeclarationStatement,
    ExpressionStatement,
    ForAllStatement,
    ReturnStatement,
)

from fhy.utils import Stack
from copy import copy

# class ASTNodeBuilder(ABC):
#     # TODO Jason: Add docstring
#     pass


# class ASTModuleBuilder(ASTNodeBuilder):
#     # TODO Jason: Add docstring
#     _components: List[Component]

#     def __init__(self) -> None:
#         self._components = []

#     @property
#     def module(self) -> Module:
#         return Module(components=self._components.copy())

#     def append_component(self, component: ASTNode) -> None:
#         self._components.append(component)


# class ASTComponentBuilder(ASTNodeBuilder):
#     @property
#     @abstractmethod
#     def component(self) -> Component:
#         ...


# class ASTFunctionBuilder(ASTComponentBuilder, ABC):
#     _name_hint: str
#     _args: List[Argument]

#     def __init__(self) -> None:
#         self._name_hint = ""
#         self._args = []

#     def set_name_hint(self, name_hint: str) -> None:
#         self._name_hint = name_hint

#     def append_arg(self, argument: Argument) -> None:
#         self._args.append(argument)


# class ASTProcedureBuilder(ASTFunctionBuilder):
#     # TODO Jason: Add docstring
#     _body: List[Statement]

#     def __init__(self) -> None:
#         super().__init__()
#         self._body = []

#     @property
#     def component(self) -> Component:
#         return Procedure(Identifier(self._name_hint), self._args, self._body)

#     def append_statement(self) -> None:
#         pass


# class ASTArgumentBuilder(ASTNodeBuilder):
#     # TODO Jason: Add docstring
#     _name_hint: str
#     _qualified_type: Optional[QualifiedType]

#     def __init__(self) -> None:
#         self._name_hint = ""
#         self._qualified_type = None

#     @property
#     def argument(self) -> Argument:
#         if self._qualified_type is None:
#             raise Exception("Qualified type must be set before creating an argument")
#         return Argument(Identifier(self._name_hint), self._qualified_type)

#     def set_name_hint(self, name_hint: str) -> None:
#         self._name_hint = name_hint

#     def set_qualified_type(self, qualified_type: QualifiedType) -> None:
#         self._qualified_type = qualified_type


# class ASTQualifiedTypeBuilder(ASTNodeBuilder):
#     # TODO Jason: Add docstring
#     _base_type: Optional[Type]
#     _type_qualifier: Optional[TypeQualifier]

#     def __init__(self) -> None:
#         self._base_type = None
#         self._type_qualifier = None

#     @property
#     def qualified_type(self) -> QualifiedType:
#         if self._base_type is None:
#             raise Exception("Base type must be set before creating a qualified type")
#         if self._type_qualifier is None:
#             raise Exception("Type qualifier must be set before creating a qualified type")
#         return QualifiedType(self._base_type, self._type_qualifier)

#     def set_base_type(self, base_type: Type) -> None:
#         self._base_type = base_type

#     def set_type_qualifier(self, type_qualifier: TypeQualifier) -> None:
#         self._type_qualifier = type_qualifier


# class ASTTypeBuilder(ASTNodeBuilder, ABC):
#     # TODO Jason: Add docstring
#     @property
#     @abstractmethod
#     def type(self) -> Type:
#         ...


# class ASTNumericalTypeBuilder(ASTNodeBuilder):
#     # TODO Jason: Add docstring
#     _primitive_data_type_name: str
#     _shape: List[Expression]

#     def __init__(self) -> None:
#         self._primitive_data_type_name = ""
#         self._shape = []

#     @property
#     def type(self) -> Type:
#         if self._primitive_data_type_name == "":
#             raise Exception("Primitive data type name must be set before creating a numerical type")
#         return NumericalType(DataType(PrimitiveDataType(self._primitive_data_type_name)), self._shape)

#     def set_primitive_data_type_name(self, primitive_data_type_name: str) -> None:
#         self._primitive_data_type_name = primitive_data_type_name

#     def append_shape(self, shape: Expression) -> None:
#         self._shape.append(shape)


# class ASTIndexTypeBuilder(ASTNodeBuilder):
#     # TODO Jason: Add docstring
#     _lower_bound: Optional[Expression]
#     _upper_bound: Optional[Expression]
#     _stride: Optional[Expression]

#     def __init__(self) -> None:
#         self._lower_bound = None
#         self._upper_bound = None
#         self._stride = None

#     @property
#     def type(self) -> Type:
#         if self._lower_bound is None:
#             raise Exception("Lower bound must be set before creating an index type")
#         if self._upper_bound is None:
#             raise Exception("Upper bound must be set before creating an index type")
#         return IndexType(self._lower_bound, self._upper_bound, self._stride)

#     def set_lower_bound(self, lower_bound: Expression) -> None:
#         self._lower_bound = lower_bound

#     def set_upper_bound(self, upper_bound: Expression) -> None:
#         self._upper_bound = upper_bound

#     def set_stride(self, stride: Expression) -> None:
#         self._stride = stride


def validate(name: str, enumeration: StrEnum) -> StrEnum:
    """Retrieves the Value from a Defined String Enumeration, and performs simple checks
    to validate the value.

    Args:
        name (str): the key name of the enumeration
        enumeration (StrEnum): An Enumeration of Supported Values

    Returns:
        (StrEnum) An instance of the provided enumeration key name.

    Raises:
        ValueError: When the name is not found within the enumeration, or is a `_PLACEHOLDER`

    """
    try:
        value = enumeration(name)
    except ValueError:
        raise ValueError(f"Unsupported {enumeration.__name__}: {name}")  # type: ignore[attr-defined]

    return value


class ContextError(Exception):
    """Unexpected Context Error"""

    @classmethod
    def message(cls, context: str, node: str, obj: Union[ASTNode, Type]) -> "ContextError":
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
MockQualifiedType = QualifiedType(base_type=_MockType, type_qualifier=_MockQual)


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
        self._node_stack.push(Module())

    def close_module_building(self) -> None:
        module_node: ASTNode = self._node_stack.pop()
        if not isinstance(module_node, Module):
            # TODO Jason: Improve exception
            raise ContextError.message("module", Module.keyname(), module_node)

        self._ast = module_node

    def add_procedure(self, name: str) -> None:
        self._node_stack.push(Procedure(name=Identifier(name)))

    def add_operation(self, name: str) -> None:
        self._node_stack.push(Operation(
            name=Identifier(name), 
            ret_type=copy(MockQualifiedType)
            ))

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

    def add_argument(self, arg_name: str) -> None:
        self._node_stack.push(Argument(name=Identifier(arg_name)))

    def close_argument_building(self) -> None:
        argument_node: ASTNode = self._node_stack.pop()
        if not isinstance(argument_node, Argument):
            raise ContextError.message("argument", Argument.keyname(), argument_node)

        function_node: ASTNode = self._node_stack.pop()
        if not isinstance(function_node, Function):
            raise ContextError.message("argument", Function.keyname(), function_node)

        args: List[Argument] = function_node.args[:]  # type: ignore[attr-defined]
        args.append(argument_node)
        new_function_node: Component = replace(function_node, args=args) # type: ignore[call-arg]
        self._node_stack.push(new_function_node)

    def add_qualified_type(self, name: str) -> None:
        qualified_type: TypeQualifier = validate(name, TypeQualifier)  # type: ignore

        # TODO: We are pushing a Type onto the stack, instead of an AST Node
        #       Can we / Should we Instead modify the previous node?
        self._node_stack.push(QualifiedType(base_type=_MockType, type_qualifier=qualified_type))

    def close_qualified_type_building(self) -> None:
        qualified_type_node: ASTNode = self._node_stack.pop()
        if not isinstance(qualified_type_node, QualifiedType):
            raise ContextError.message("qualified type", QualifiedType.keyname(), qualified_type_node)

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
                raise NotImplementedError(f"Other Statements Not Implemented Yet: {previous_node}")

        else:
            raise ContextError.message(
                "qualified type", f"({Argument.keyname()} | {Operation.keyname()})", previous_node
                )

        new_node: ASTNode = replace(previous_node, **kwargs)  # type: ignore[arg-type]
        self._node_stack.push(new_node)

    def add_dtype(self, dtype):
        node: ASTNode = self.get_current_node()

        if not isinstance(node, Type):
            raise ContextError(f"Adding dtype. Current Node is not of type `Type`. Received: {node}")

        data_type: PrimitiveDataType = validate(dtype, PrimitiveDataType)
        node._data_type = DataType(data_type)

    def add_numerical_type(self) -> None:
        self._node_stack.push(NumericalType(_MockType, []))

    def add_shape(self, shapes: List[str]):
        node: Type = self.get_current_node()
        if not isinstance(node, Type):
            raise ContextError(f"Adding Shape. Current Node is not of type `Type`. Received: {node}")
        if not hasattr(node, "_shape") or not isinstance(node._shape, list):
            node._shape = []

        # NOTE: We Support Mixed Identification of Shape. Consider the
        #       Following Variations: [n, m] vs [2, 4] vs [2, n]
        # TODO: Consider and Implement this Variant: [n + m, n - 1]
        for s in shapes:
            if s.isnumeric():
                obj = IntLiteral(int(s))
            else:
                obj = IdentifierExpression(Identifier(s))
            node._shape.append(obj)

    def add_index_type(self) -> None:
        self._node_stack.push(IndexType(None, None))

    def close_type_building(self) -> None:
        type_node: Type = self._node_stack.pop()
        if not isinstance(type_node, Type):
            raise ContextError.message("type", "Type", type_node)

        qualified_type_node: QualifiedType = self.get_current_node()
        if not isinstance(qualified_type_node, QualifiedType):
            raise ContextError.message("type", "QualifiedType", qualified_type_node)

        qualified_type_node._base_type = type_node

    def open_declaration_statement(self, name: str):
        node = DeclarationStatement(
            _variable_name=Identifier(name),
            _variable_type=MockQualifiedType,
        )
        self._node_stack.push(node)
 
    def close_declaration_statement(self):
        express: ASTNode = self._node_stack.pop()

        # Declaration Statment May Not Assign a Value.
        if isinstance(express, DeclarationStatement):
            self._node_stack.push(express)
            return

        if not isinstance(express, Expression):
            raise ContextError.message("declaration_statement", Expression.keyname(), express)

        current: ASTNode = self._node_stack.pop()
        if not isinstance(current, DeclarationStatement):
            raise ContextError.message("declaration_statement", DeclarationStatement.keyname(), current)

        new_node = replace(current, _expression=express)
        self._node_stack.push(new_node)

    def open_branch_statement(self):
        node = BranchStatement(Expression, [], [])
        self._node_stack.push(node)

    def open_iteration_statement(self):
        node = ForAllStatement(Expression, [])
        self._node_stack.push(node)

    def open_return_statement(self):
        node = ReturnStatement(_expression=Expression)
        self._node_stack.push(node)

    def close_return_statement(self):
        express: ASTNode = self._node_stack.pop()

        # Return Statement May Not Have a Value.
        if isinstance(express, ReturnStatement):
            self._node_stack.push(express)
            return

        if not isinstance(express, Expression):
            raise ContextError.message("return_statement", Expression.keyname(), express)

        current: ASTNode = self._node_stack.pop()
        if not isinstance(current, ReturnStatement):
            raise ContextError.message("return_statement", ReturnStatement.keyname(), current)

        new_node = replace(current, _expression=express)
        self._node_stack.push(new_node)

    def close_statement(self):
        # NOTE: This is a General Close of Any Statement. 
        #       Each Statement Type Needs it's Own Prep to Close Correctly.
        _statement: ASTNode = self._node_stack.pop()
        if not isinstance(_statement, Statement):
            raise ContextError.message("statement", Statement.keyname(), _statement)

        current: ASTNode = self._node_stack.pop()
        if not isinstance(current, Function):
            raise ContextError.message("statement", Function.keyname(), current)

        body: List[Statement] = []
        if hasattr(current, "body") and current.body is not None:
            body.extend(current.body)
        body.append(_statement)

        new_node: Function = replace(current, body=body)
        self._node_stack.push(new_node)

    def add_literal(self, value: Union[int, float, complex]):
        if isinstance(value, complex):
            node = ComplexLiteral(value)
        elif isinstance(value, float):
            node = FloatLiteral(value)
        elif isinstance(value, int):
            node = IntLiteral(value)
        else:
            raise NotImplementedError("Unknown Literal")
        self._node_stack.push(node)

    def add_unary_expression(self, operator: str):
        op = validate(operator, UnaryOperation)
        node = UnaryExpression(_operation=op, _expression=None)
        self._node_stack.push(node)
    
    def close_unary_expression(self):
        expression_node: ASTNode = self._node_stack.pop()
        if not isinstance(expression_node, Expression):
            raise ContextError.message("unary_expression", Expression.keyname(), expression_node)
    
        current: ASTNode = self.get_current_node()
        if not isinstance(current, UnaryExpression):
            raise ContextError.message("unary_expression", UnaryExpression.keyname(), current)

        current._expression = expression_node

    def add_binary_expression(self, operator: str):
        op = validate(operator, BinaryOperation)
        node = BinaryExpression(
            _operation=op, 
            _left_expression=None, 
            _right_expression=None
        )

        self._node_stack.push(node)

    def close_binary_expression(self):
        right: ASTNode = self._node_stack.pop()
        if not isinstance(right, Expression):
            raise ContextError.message("binary_expression", Expression.keyname(), right)

        left: ASTNode = self._node_stack.pop()
        if not isinstance(left, Expression):
            raise ContextError.message("binary_expression", Expression.keyname(), left)

        binary: ASTNode = self.get_current_node()
        if not isinstance(binary, BinaryExpression):
            raise ContextError.message("binary_expression", BinaryExpression.keyname(), binary)

        binary._left_expression = left
        binary._right_expression = right

    def open_ternary_expression(self):
        node = TernaryExpression(
            _condition=Expression,
            _true_expression=Expression,
            _false_expression=Expression
            )
        self._node_stack.push(node)

    def close_ternary_expression(self):
        _false = self._node_stack.pop()
        if not isinstance(_false, Expression):
            raise ContextError.message("ternary_expression", f"false {Expression.keyname()}", _false)

        _true = self._node_stack.pop()
        if not isinstance(_true, Expression):
            raise ContextError.message("ternary_expression", f"true {Expression.keyname()}", _true)

        _condition = self._node_stack.pop()
        if not isinstance(_condition, Expression):
            raise ContextError.message("ternary_expression", f"condition {Expression.keyname()}", _condition)

        current = self.get_current_node()
        if not isinstance(current, TernaryExpression):
            raise ContextError.message("ternary_expression", TernaryExpression.keyname(), current)

        current._condition = _condition
        current._true_expression = _true
        current._false_expression = _false
