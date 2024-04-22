""" """
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, Field, MISSING
from typing import Any, Dict, List, Set, Type, Optional, Union

from fhy import ir

from ..ast import ASTNode, Expression
from ..ast.directory import get_ast_node_type_info, ASTNodeTypeInfo


class FieldAttributeError(Exception):
    """Raised when Attempt to Assign an Unsupported Attribute."""


class ASTBuilderFrame(ABC):
    _cls: type

    def __init__(self, cls: type) -> None:
        self._cls = cls

    @property
    def cls(self) -> type:
        return self._cls

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def build(self) -> Union[ASTNode, ir.Type]:
        ...


class ASTNodeBuilderFrame(ASTBuilderFrame):
    """Core Builder Class Constructor.

    Args:
        node_class (Type[ASTNode]): Registered ASTNode cls
        **kwargs: Supported Keyword Arguments of the ASTNode

    Raises:
        UnregisteredASTNode: When Provided
        FieldAttributeError:

    """
    _type_info: ASTNodeTypeInfo

    def __init__(self, cls: Type[ASTNode], **kwargs: Any) -> None:
        super().__init__(cls)
        self._type_info: ASTNodeTypeInfo = get_ast_node_type_info(self.cls)

        for key in self._node_annotations.keys():
            if not hasattr(self, key):
                if key in kwargs:
                    setattr(self, key, kwargs[key])
                elif hasattr(self.cls, "__dataclass_fields__") and key in getattr(self.cls, "__dataclass_fields__"):
                    field: Any = getattr(self.cls, "__dataclass_fields__")[key]
                    assert isinstance(field, Field), f"Dataclass field {key} is not a dataclasses Field object"
                    if field.default is not MISSING:
                        setattr(self, key, field.default)
                    elif field.default_factory is not MISSING:
                        setattr(self, key, field.default_factory())
                    else:
                        setattr(self, key, None)

    @property
    def _node_annotations(self) -> Dict[str, type]:
        return self._type_info.fields

    def build(self) -> Union[ASTNode, ir.Type]:
        """Builds the Desired Class using the collected attributes."""
        data = {k: getattr(self, k, None) for k in self._node_annotations.keys()}
        return self.cls(**data)

    def update(self, **kwargs: Any) -> None:
        """Updates and/or Overwrites the Class Attributes"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in getattr(super(), "__annotations__").keys() and name not in getattr(self, "__annotations__").keys() and name not in self._node_annotations.keys():
            raise FieldAttributeError(f"Unsupported Attribute Assignment: {name}")
        super().__setattr__(name, value)


class _TypeInfo(ABC):
    @abstractmethod
    def build_type(self) -> ir.Type:
        ...

    @abstractmethod
    def update(self, **kwargs: Any):
        ...


@dataclass
class _NumericalTypeInfo(_TypeInfo):
    data_type: Optional[ir.DataType] = field(default=None)
    shape: Optional[List[Expression]] = field(default=None)

    def build_type(self) -> ir.NumericalType:
        if self.data_type is None:
            raise Exception()
        if self.shape is None:
            raise Exception()
        return ir.NumericalType(data_type=self.data_type, shape=self.shape)

    def update(self, **kwargs: Any):
        if "data_type" in kwargs:
            self.data_type = kwargs["data_type"]
        if "shape" in kwargs:
            self.shape = kwargs["shape"]


@dataclass
class _IndexTypeInfo(Type):
    lower_bound: Optional[Expression] = field(default=None)
    upper_bound: Optional[Expression] = field(default=None)
    stride: Optional[Expression] = field(default=None)

    def build_type(self) -> ir.IndexType:
        if self.lower_bound is None:
            raise Exception()
        if self.upper_bound is None:
            raise Exception()
        return ir.IndexType(lower_bound=self.lower_bound, upper_bound=self.upper_bound, stride=self.stride)

    def update(self, **kwargs: Any) -> None:
        if "lower_bound" in kwargs:
            self.lower_bound = kwargs["lower_bound"]
        if "upper_bound" in kwargs:
            self.upper_bound = kwargs["upper_bound"]
        if "stride" in kwargs:
            self.stride = kwargs["stride"]


class TypeBuilderFrame(ASTBuilderFrame):
    _type_info: Union[_TypeInfo]

    def __init__(self, cls: Type[ir.Type], **kwargs: Any) -> None:
        super().__init__(cls)
        if issubclass(cls, ir.NumericalType):
            self._type_info = _NumericalTypeInfo()
        elif issubclass(cls, ir.IndexType):
            self._type_info = _IndexTypeInfo()
        else:
            raise Exception()

    def build(self) -> Union[ASTNode, ir.Type]:
        return self._type_info.build_type()

    def update(self, **kwargs: Any) -> None:
        self._type_info.update(**kwargs)


def create_builder_frame(cls: type, **kwargs: Any) -> ASTBuilderFrame:
    if issubclass(cls, ASTNode):
        return ASTNodeBuilderFrame(cls, **kwargs)
    elif issubclass(cls, ir.Type):
        return TypeBuilderFrame(cls, **kwargs)
    else:
        raise Exception()
