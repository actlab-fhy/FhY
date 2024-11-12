# Copyright (c) 2024 FhY Developers
# Christopher Priebe <cpriebe@ucsd.edu>
# Jason C Del Rio <j3delrio@ucsd.edu>
# Hadi S Esmaeilzadeh <hadi@ucsd.edu>
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be
# used to endorse or promote products derived from this software without specific prior
# written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""Define type aliases, or generic types describing core FhY language constructs."""

from typing import Generic, TypeVar

from fhy_core import (
    CoreDataType,
    DataType,
    Identifier,
    PrimitiveDataType,
    TemplateDataType,
    Type,
    TypeQualifier,
)
from fhy_core import Expression as CoreExpression

from fhy.lang.ast.node import ASTNode, Expression, Function, Statement
from fhy.lang.ast.span import Source, Span

ASTTypeNodes = ASTNode | Type
Nodes = TypeVar("Nodes", bound=ASTTypeNodes)

Expressions = CoreExpression | Expression
ExpressionNodes = TypeVar("ExpressionNodes", bound=Expressions)

CoreASTNodes = Statement | Function | Expression
Core = TypeVar("Core", bound=CoreASTNodes)

OtherTypes = (
    Identifier
    | DataType
    | TypeQualifier
    | PrimitiveDataType
    | TemplateDataType
    | CoreDataType
)

Spans = Span | Source

_ASTObject = ASTTypeNodes | Expressions | CoreASTNodes | OtherTypes | Spans
ASTObject = TypeVar("ASTObject", bound=_ASTObject)


class ASTNodes(Generic[ASTObject]):
    """Bound generic AST node."""
