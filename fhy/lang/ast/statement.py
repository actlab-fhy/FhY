"""All ASTNodes defined within this module are Subclasses of the core.Statement ASTNode

Statement ASTNodes:
    DeclarationStatement: Declares a Variable, with or without assignment
    ExpressionStatement:
    ForAllStatement: An Iteration statement evaluating an expression over a body

"""

from dataclasses import dataclass, field
from typing import List, Optional

from .core import Statement
from .expression import Expression, Identifier
from .qualified_type import QualifiedType


@dataclass(frozen=True, kw_only=True)
class DeclarationStatement(Statement):
    """Declaration Statements Are Declaration or Assignment to a Variable Name.

    Args:
        variable_name (Identifier): identity of a variable
        variable_type (QualifiedType): qualified type of variable
        expression (Optional[Expression]): optional expression assignment to variable

    """

    variable_name: Identifier
    variable_type: QualifiedType
    expression: Optional[Expression] = field(default=None)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["variable_name", "variable_type", "expression"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class ExpressionStatement(Statement):
    """Expression Statement"""

    left: Optional[Expression] = field(default=None)
    right: Expression

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["left", "right"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class ForAllStatement(Statement):
    """A Statement ASTNode describing a for loop.

    Args:
        index (Expression): Expression to be iterated through
        body (List[Statement]): Body of for loop statements to be performed iteratively

    """

    index: Expression
    body: List[Statement] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["index", "body"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class SelectionStatement(Statement):
    """A Conditional Branch Statement ASTNode.

    Args:
        condition (Expression): Condition to be evaluated
        true_body (List[Statement]): Body of Statements Evaluated if True
        false_body (List[Statement]): Body of Statements Evaluated if False

    """

    condition: Expression
    true_body: List[Statement] = field(default_factory=list)
    false_body: List[Statement] = field(default_factory=list)

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["condition", "true_body", "false_body"])
        return attrs


@dataclass(frozen=True, kw_only=True)
class ReturnStatement(Statement):
    """A control flow return Statement ASTNode.

    Args:
        expression (Expression): expression value to be returned

    """

    expression: Expression

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["expression"])
        return attrs
