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

"""Core AST nodes for FhY language constructs.

Core Nodes:
    Module: Module node.

Core Abstract Nodes:
    Function: Base function node.
    Statement: Base statement node.
    Expression: Base AST expression node.

"""

from abc import ABC
from dataclasses import dataclass, field
from typing import List

from fhy.ir.expression import Expression as IRExpression
from fhy.ir.identifier import Identifier as IRIdentifier

from .base import ASTNode


@dataclass(frozen=True, kw_only=True)
class Module(ASTNode):
    """FhY module AST node.

    Args:
        name (IRIdentifier): Name of the module.
        statements (List[Statement]): List of statements in the module.

    Attributes:
        name (IRIdentifier): Name of the module.
        statements (List[Statement]): List of statements in the module.

    """

    # TODO: remove default value for name and have converter create name
    name: IRIdentifier = field(default=IRIdentifier("module"))
    statements: List["Statement"] = field(default_factory=list)

    def get_visit_attrs(self) -> List[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["name", "statements"])
        return attrs


class Statement(ASTNode, ABC):
    """Abstract statement AST node."""


@dataclass(frozen=True, kw_only=True)
class Function(Statement, ABC):
    """Abstract FhY function node.

    Used as a base for the function nodes such as procedures and operations.

    Attributes:
        name (IRIdentifier): Name of the function.

    """

    name: IRIdentifier

    def get_visit_attrs(self) -> List[str]:
        attrs = super().get_visit_attrs()
        attrs.extend(["name"])
        return attrs


class Expression(ASTNode, IRExpression, ABC):
    """Abstract expression AST node.

    Also is an expression from the IR to enable use in symbol table fields.

    """
