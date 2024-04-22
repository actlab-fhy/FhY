# TODO Jason: Add docstring
from dataclasses import dataclass, field
from typing import List, Optional

from .core import Statement
from .expression import Expression, Identifier
from .qualified_type import QualifiedType


@dataclass(frozen=True, kw_only=True)
class DeclarationStatement(Statement):
    """Declaration Statements Are Declaration or Assignment to a Variable Name.

    Args:
        _variable_name (Identifier):
        _variable_type (QualifiedType):
        _expression (Optional[Expression]):

    """

    variable_name: Identifier
    variable_type: QualifiedType
    expression: Optional[Expression] = field(default=None)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_variable_name", "_variable_type", "_expression"])
        return attrs

    # TODO Jason: Implement the functionality of this class


@dataclass(frozen=True, kw_only=True)
class ExpressionStatement(Statement):
    """Expression Statement"""

    # TODO Jason: Add docstring
    left: Optional[Expression] = field(default=None)
    right: Expression

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_left", "_index", "_right"])
        return attrs

    # TODO Jason: Implement the functionality of this class


@dataclass(frozen=True, kw_only=True)
class ForAllStatement(Statement):
    """For Loop Node"""

    index: Expression
    body: List[Statement] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["index", "body"])
        return attrs

    # TODO Jason: Implement the functionality of this class


@dataclass(frozen=True, kw_only=True)
class SelectionStatement(Statement):
    """A Branch (Conditional) Statement Block Node.

    Args:
        _predicate (Expression): Condition to Be Evaluated
        _true_body (List[Statement]): Body of Statements Evaluated if True
        _false_body (List[Statement]): Body of Statements Evaluated if False

    """

    condition: Expression
    true_body: List[Statement] = field(default_factory=list)
    false_body: List[Statement] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["predicate", "true_body", "false_body"])
        return attrs

    # TODO Jason: Implement the functionality of this class


@dataclass(frozen=True, kw_only=True)
class ReturnStatement(Statement):
    expression: Expression

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["expression"])
        return attrs
