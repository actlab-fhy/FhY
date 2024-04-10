# TODO Jason: Add docstring
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from fhy.lang.ast import Argument, ASTNode, Component, DataType, Expression, Identifier, IndexType, Module, NumericalType, PrimitiveDataType, Procedure, QualifiedType, Statement, Type, TypeQualifier
from fhy.utils import Stack


class ASTNodeBuilder(ABC):
    # TODO Jason: Add docstring
    pass


class ASTModuleBuilder(ASTNodeBuilder):
    # TODO Jason: Add docstring
    _components: List[Component]

    def __init__(self) -> None:
        self._components = []

    @property
    def module(self) -> Module:
        return Module(*self._components)

    def append_component(self, component: ASTNode) -> None:
        self._components.append(component)


class ASTComponentBuilder(ASTNodeBuilder):
    @property
    @abstractmethod
    def component(self) -> Component:
        ...


class ASTFunctionBuilder(ASTComponentBuilder, ABC):
    _name_hint: str
    _args: List[Argument]

    def __init__(self) -> None:
        self._name_hint = ""
        self._args = []

    def set_name_hint(self, name_hint: str) -> None:
        self._name_hint = name_hint

    def append_arg(self, argument: Argument) -> None:
        self._args.append(argument)


class ASTProcedureBuilder(ASTFunctionBuilder):
    # TODO Jason: Add docstring
    _body: List[Statement]

    def __init__(self) -> None:
        super().__init__()
        self._body = []

    @property
    def component(self) -> Component:
        return Procedure(Identifier(self._name_hint), self._args, self._body)

    def append_statement(self) -> None:
        pass


class ASTArgumentBuilder(ASTNodeBuilder):
    # TODO Jason: Add docstring
    _name_hint: str
    _qualified_type: Optional[QualifiedType]

    def __init__(self) -> None:
        self._name_hint = ""
        self._qualified_type = None

    @property
    def argument(self) -> Argument:
        if self._qualified_type is None:
            raise Exception("Qualified type must be set before creating an argument")
        return Argument(Identifier(self._name_hint), self._qualified_type)

    def set_name_hint(self, name_hint: str) -> None:
        self._name_hint = name_hint

    def set_qualified_type(self, qualified_type: QualifiedType) -> None:
        self._qualified_type = qualified_type


class ASTQualifiedTypeBuilder(ASTNodeBuilder):
    # TODO Jason: Add docstring
    _base_type: Optional[Type]
    _type_qualifier: Optional[TypeQualifier]

    def __init__(self) -> None:
        self._base_type = None
        self._type_qualifier = None

    @property
    def qualified_type(self) -> QualifiedType:
        if self._base_type is None:
            raise Exception("Base type must be set before creating a qualified type")
        if self._type_qualifier is None:
            raise Exception("Type qualifier must be set before creating a qualified type")
        return QualifiedType(self._base_type, self._type_qualifier)

    def set_base_type(self, base_type: Type) -> None:
        self._base_type = base_type

    def set_type_qualifier(self, type_qualifier: TypeQualifier) -> None:
        self._type_qualifier = type_qualifier


class ASTTypeBuilder(ASTNodeBuilder, ABC):
    # TODO Jason: Add docstring
    @property
    @abstractmethod
    def type(self) -> Type:
        ...


class ASTNumericalTypeBuilder(ASTNodeBuilder):
    # TODO Jason: Add docstring
    _primitive_data_type_name: str
    _shape: List[Expression]

    def __init__(self) -> None:
        self._primitive_data_type_name = ""
        self._shape = []

    @property
    def type(self) -> Type:
        if self._primitive_data_type_name == "":
            raise Exception("Primitive data type name must be set before creating a numerical type")
        return NumericalType(DataType(PrimitiveDataType(self._primitive_data_type_name)), self._shape)

    def set_primitive_data_type_name(self, primitive_data_type_name: str) -> None:
        self._primitive_data_type_name = primitive_data_type_name

    def append_shape(self, shape: Expression) -> None:
        self._shape.append(shape)


class ASTIndexTypeBuilder(ASTNodeBuilder):
    # TODO Jason: Add docstring
    _lower_bound: Optional[Expression]
    _upper_bound: Optional[Expression]
    _stride: Optional[Expression]

    def __init__(self) -> None:
        self._lower_bound = None
        self._upper_bound = None
        self._stride = None

    @property
    def type(self) -> Type:
        if self._lower_bound is None:
            raise Exception("Lower bound must be set before creating an index type")
        if self._upper_bound is None:
            raise Exception("Upper bound must be set before creating an index type")
        return IndexType(self._lower_bound, self._upper_bound, self._stride)

    def set_lower_bound(self, lower_bound: Expression) -> None:
        self._lower_bound = lower_bound

    def set_upper_bound(self, upper_bound: Expression) -> None:
        self._upper_bound = upper_bound

    def set_stride(self, stride: Expression) -> None:
        self._stride = stride


class ASTBuilder(object):
    # TODO Jason: Add docstring
    _builder_stack: Stack[ASTNodeBuilder]
    _ast: Optional[ASTNode]

    def __init__(self):
        self._builder_stack = Stack[ASTNodeBuilder]()
        self._ast = None

    @property
    def ast(self) -> Optional[ASTNode]:
        return self._ast

    def get_current_builder(self) -> Optional[ASTNodeBuilder]:
        if len(self._builder_stack) == 0:
            return None
        else:
            return self._builder_stack.peek()

    def open_module_context(self) -> None:
        self._builder_stack.push(ASTModuleBuilder())

    def close_module_context(self) -> None:
        module_builder: ASTNodeBuilder = self._builder_stack.pop()
        if not isinstance(module_builder, ASTModuleBuilder):
            # TODO Jason: Improve exception
            raise Exception("Cannot close module builder context when not in module context")
        self._ast = module_builder.module

    def open_procedure_context(self) -> None:
        self._builder_stack.push(ASTProcedureBuilder())

    def close_component_context(self) -> None:
        component_builder: ASTNodeBuilder = self._builder_stack.pop()
        module_builder: ASTNodeBuilder = self.get_current_builder()
        if not isinstance(component_builder, ASTComponentBuilder) and not isinstance(module_builder, ASTModuleBuilder):
            # TODO Jason: Improve exception
            raise Exception("Cannot close component builder context when not in component context")
        module_builder.append_component(component_builder.component)

    def open_argument_context(self) -> None:
        self._builder_stack.push(ASTArgumentBuilder())

    def close_argument_context(self) -> None:
        argument_builder: ASTNodeBuilder = self._builder_stack.pop()
        function_builder: ASTNodeBuilder = self.get_current_builder()
        if not isinstance(argument_builder, ASTArgumentBuilder) and not isinstance(function_builder, ASTFunctionBuilder):
            # TODO Jason: Improve exception
            raise Exception("Cannot close argument builder context when not in argument context")
        function_builder.append_arg(argument_builder.argument)

    def open_qualified_type_context(self) -> None:
        self._builder_stack.push(ASTQualifiedTypeBuilder())

    def close_qualified_type_context(self) -> None:
        qualified_type_builder: ASTNodeBuilder = self._builder_stack.pop()
        argument_builder: ASTNodeBuilder = self.get_current_builder()
        if not isinstance(qualified_type_builder, ASTQualifiedTypeBuilder) and not isinstance(argument_builder, ASTArgumentBuilder):
            # TODO Jason: Improve exception
            raise Exception("Cannot close qualified type builder context when not in qualified type context")
        argument_builder.set_qualified_type(qualified_type_builder.qualified_type)

    def open_numerical_type_context(self) -> None:
        self._builder_stack.push(ASTNumericalTypeBuilder())

    def open_index_type_context(self) -> None:
        self._builder_stack.push(ASTIndexTypeBuilder())

    def close_type_context(self) -> None:
        type_builder: ASTNodeBuilder = self._builder_stack.pop()
        qualified_type_builder: ASTNodeBuilder = self.get_current_builder()
        if not isinstance(type_builder, ASTTypeBuilder) and not isinstance(qualified_type_builder, ASTQualifiedTypeBuilder):
            # TODO Jason: Improve exception
            raise Exception("Cannot close type builder context when not in type context")
        qualified_type_builder.set_base_type(type_builder.type)
