# TODO Jason: Add docstring
from dataclasses import replace
from enum import StrEnum
from typing import List, Optional, Union
from fhy.lang.ast import Argument, ASTNode, Component, Expression, Function, Module, Procedure, QualifiedType, Statement
from fhy.ir import DataType, Identifier, IndexType, NumericalType, PrimitiveDataType, Type, TypeQualifier
from fhy.utils import Stack


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
    try:
        val: StrEnum = enumeration[name.upper()]
        assert val.value == name  # Make Keyword Case Sensitive
    except (KeyError, AssertionError):
        raise ValueError(f"Unsupported {enumeration.__name__}: {name}")

    return val


class ASTBuilder(object):
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
            raise Exception("Cannot close module context when not in module context")
        self._ast = module_node

    def add_procedure(self, procedure_name: str) -> None:
        self._node_stack.push(Procedure(name=Identifier(procedure_name)))

    def close_component_building(self) -> None:
        component_node: ASTNode = self._node_stack.pop()
        module_node: ASTNode = self._node_stack.pop()
        if not isinstance(component_node, Component) and not isinstance(module_node, Module):
            # TODO Jason: Improve exception
            raise Exception("Cannot close component context when not in component context")
        components: List[Component] = module_node.components
        components.append(component_node)
        new_module_node: Module = replace(module_node, components=components)
        self._node_stack.push(new_module_node)

    def add_argument(self, arg_name: str) -> None:
        self._node_stack.push(Argument(name=Identifier(arg_name)))

    def close_argument_building(self) -> None:
        argument_node: ASTNode = self._node_stack.pop()
        function_node: ASTNode = self._node_stack.pop()
        if not isinstance(argument_node, Argument) and not isinstance(function_node, Function):
            # TODO Jason: Improve exception
            raise Exception("Cannot close argument builder context when not in argument context")
        args: List[Argument] = function_node.args[:]
        args.append(argument_node)
        new_function_node: Component = replace(function_node, args=args)
        self._node_stack.push(new_function_node)

    def add_qualified_type(self, name: str) -> None:
        qualified_type: TypeQualifier = validate(name, TypeQualifier)
        self._node_stack.push(QualifiedType(type_qualifier=qualified_type))

    def close_qualified_type_building(self) -> None:
        qualified_type_node: ASTNode = self._node_stack.pop()
        argument_node: ASTNode = self._node_stack.pop()
        if not isinstance(qualified_type_node, QualifiedType) and not isinstance(argument_node, Argument):
            # TODO Jason: Improve exception
            raise Exception(
                f"Cannot close Qualified Type: {qualified_type_node} | Argument Type: {argument_node}"
                )

        new_argument_node: Argument = replace(argument_node, qualified_type=qualified_type_node)
        self._node_stack.push(new_argument_node)

    def add_dtype(self, dtype):
        node: ASTNode = self.get_current_node()
        if not isinstance(node, Type):
            raise Exception("Node is not of type Type")

        data_type: PrimitiveDataType = validate(dtype, PrimitiveDataType)
        node._primitive_data_type = DataType(data_type)

    def add_numerical_type(self) -> None:
        self._node_stack.push(NumericalType(None, []))

    def add_index_type(self) -> None:
        self._node_stack.push(IndexType(None, None))

    def close_type_building(self) -> None:
        type_node: Type = self._node_stack.pop()
        qualified_type_node: QualifiedType = self.get_current_node()
        if not isinstance(type_node, Type) and not isinstance(qualified_type_node, QualifiedType):
            # TODO Jason: Improve exception
            raise Exception("Cannot close type node context when not in type context")
        qualified_type_node._base_type = type_node
