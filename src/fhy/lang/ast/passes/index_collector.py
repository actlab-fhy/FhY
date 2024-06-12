from typing import Any, Callable

from fhy.ir.identifier import Identifier
from fhy.lang.ast.node.expression import FunctionExpression, IdentifierExpression
from fhy.lang.ast.visitor import Visitor
from fhy.lang.ast.node import core


class IndexCollector(Visitor):
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

    def __call__(self, node: Any) -> Any:
        if not isinstance(node, core.Expression):
            raise RuntimeError(f"{__class__.__name__} expects an expression node.")
        return super().__call__(node)

    def visit_Identifier(self, node: Identifier) -> None:
        if self._is_identifier_index(node):
            self._indices.add(node)


def collect_indices(
    node: core.Expression,
    is_identifier_index: Callable[[Identifier], bool],
) -> set[Identifier]:
    index_collector = IndexCollector(is_identifier_index)
    index_collector(node)
    return index_collector.indices


# TODO: if the language supports indices in reduction's parameters that are not just the identifier itself, this pass must be modified
class ReducedIndexCollector(Visitor):
    _is_identifier_index: Callable[[Identifier], bool]
    _reduced_indices: set[Identifier]

    def __init__(self, is_identifier_index_func: Callable[[Identifier], bool]) -> None:
        super().__init__()
        self._reduced_indices = set()
        self._is_identifier_index = is_identifier_index_func

    @property
    def reduced_indices(self) -> set[Identifier]:
        return self._reduced_indices

    def __call__(self, node: Any) -> Any:
        if not isinstance(node, core.Expression):
            raise RuntimeError(f"{__class__.__name__} expects an expression node.")
        return super().__call__(node)

    def visit_FunctionExpression(self, node: FunctionExpression) -> None:
        for index in node.indices:
            if not isinstance(index, IdentifierExpression):
                raise RuntimeError()
            if self._is_identifier_index(index.identifier):
                self._reduced_indices.add(index.identifier)

        super().visit_FunctionExpression(node)


def collect_reduced_indices(
    node: core.Expression,
    is_identifier_index: Callable[[Identifier], bool],
) -> set[Identifier]:
    reduced_index_collector = ReducedIndexCollector(is_identifier_index)
    reduced_index_collector(node)
    return reduced_index_collector.reduced_indices
