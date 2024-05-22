"""Define Type Aliases, or Generic Types describing core FhY Language Constructs."""

from typing import TypeVar, Union

from fhy import ir as _ir
from fhy.lang.ast.node import base, core
from fhy.lang.ast.span import Source, Span

# Define Commonly Used Aliases
ASTTypeNodes = Union[base.ASTNode, _ir.Type]
Nodes = TypeVar("Nodes", bound=ASTTypeNodes)

Expressions = Union[_ir.Expression, core.Expression]
ExpressionNodes = TypeVar("ExpressionNodes", bound=Expressions)

CoreASTNodes = Union[core.Statement, core.Function, core.Expression]
Core = TypeVar("Core", bound=CoreASTNodes)

OtherTypes = Union[
    _ir.Identifier, _ir.DataType, _ir.TypeQualifier, _ir.PrimitiveDataType
]
Spans = Union[Span, Source]

_ASTObject = Union[ASTTypeNodes, Expressions, CoreASTNodes, OtherTypes, Spans]
ASTObject = TypeVar("ASTObject", bound=_ASTObject)
