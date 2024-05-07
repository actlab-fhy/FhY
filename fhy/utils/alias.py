"""Define Type Aliases, or Generic Types describing core FhY Language Constructs."""

from typing import TypeVar, Union

from fhy import ir as _ir
from fhy.lang import ast as _ast
from fhy.lang.span import Source, Span

# Define Commonly Used Aliases
ASTTypeNodes = Union[_ast.ASTNode, _ir.Type]
Nodes = TypeVar("Nodes", bound=ASTTypeNodes)

Expressions = Union[_ir.Expression, _ast.Expression]
ExpressionNodes = TypeVar("ExpressionNodes", bound=Expressions)

CoreASTNodes = Union[_ast.Statement, _ast.Function, _ast.Expression]
Core = TypeVar("Core", bound=CoreASTNodes)

OtherTypes = Union[
    _ir.Identifier, _ir.DataType, _ir.TypeQualifier, _ir.PrimitiveDataType
]
Spans = Union[Span, Source]

_ASTObject = Union[ASTTypeNodes, Expressions, CoreASTNodes, OtherTypes, Spans]
ASTObject = TypeVar("ASTObject", bound=_ASTObject)
