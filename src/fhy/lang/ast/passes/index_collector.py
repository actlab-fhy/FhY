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

"""Index collection passes."""

from collections.abc import Callable

from fhy_core import Identifier

from fhy.lang.ast.node import core
from fhy.lang.ast.node.expression import FunctionExpression, IdentifierExpression
from fhy.lang.ast.visitor import ExpressionVisitor


class IndexCollector(ExpressionVisitor):
    """Collect all the indices used in an AST expression."""

    _is_identifier_index: Callable[[Identifier], bool]
    _indices: set[Identifier]

    def __init__(self, is_identifier_index_func: Callable[[Identifier], bool]) -> None:
        super().__init__()
        self._indices = set()
        self._is_identifier_index = is_identifier_index_func

    @property
    def indices(self) -> set[Identifier]:
        return self._indices

    def visit_identifier(self, node: Identifier) -> None:
        if self._is_identifier_index(node):
            self._indices.add(node)


def collect_indices(
    node: core.Expression,
    is_identifier_index: Callable[[Identifier], bool],
) -> set[Identifier]:
    """Collect all the indices used in an AST expression.

    Args:
        node (core.Expression): The AST expression node to collect indices from.
        is_identifier_index (Callable[[Identifier], bool]): A function that
            determines if an identifier is an index.

    Returns:
        set[Identifier]: The set of indices used in the AST expression.

    """
    index_collector = IndexCollector(is_identifier_index)
    index_collector(node)
    return index_collector.indices


# TODO: if the language supports indices in reduction's parameters that are not
#       just the identifier itself, this pass must be modified
class ReducedIndexCollector(ExpressionVisitor):
    """Collect all the indices used in an AST expression that are reduced."""

    _is_identifier_index: Callable[[Identifier], bool]
    _reduced_indices: set[Identifier]

    def __init__(self, is_identifier_index_func: Callable[[Identifier], bool]) -> None:
        super().__init__()
        self._reduced_indices = set()
        self._is_identifier_index = is_identifier_index_func

    @property
    def reduced_indices(self) -> set[Identifier]:
        return self._reduced_indices

    def visit_function_expression(self, node: FunctionExpression) -> None:
        for index in node.indices:
            if not isinstance(index, IdentifierExpression):
                raise RuntimeError()
            if self._is_identifier_index(index.identifier):
                self._reduced_indices.add(index.identifier)

        super().visit_function_expression(node)


def collect_reduced_indices(
    node: core.Expression,
    is_identifier_index: Callable[[Identifier], bool],
) -> set[Identifier]:
    """Collect all the indices used in an AST expression that are reduced.

    Args:
        node (core.Expression): The AST expression node to collect indices from.
        is_identifier_index (Callable[[Identifier], bool]): A function that
            determines if an identifier is an index.

    Returns:
        set[Identifier]: The set of indices used in the AST expression that
            are reduced.

    """
    reduced_index_collector = ReducedIndexCollector(is_identifier_index)
    reduced_index_collector(node)
    return reduced_index_collector.reduced_indices
