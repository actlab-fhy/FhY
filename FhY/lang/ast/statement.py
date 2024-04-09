# TODO Jason: Add docstring
from typing import List, Optional
from .expression import Identifier
from .base import Statement
from .expression import Expression
from .type import Type, TypeQualifier


class DeclarationStatement(Statement):
    """Declaration Statements Are Declaration or Assignment to a Variable Name.

    Args:
        _variable_name (Identifier):
        _variable_type (Type):
        _variable_type_qual (TypeQualifier): 
        _expression (Optional[Expression]):

    """

    def __init__(self,
                 _variable_name: Identifier,
                 _variable_type: Type,
                 _variable_type_qual: TypeQualifier,
                 _expression: Optional[Expression] = None
                 ) -> None:
        super().__init__()
        self._variable_name = _variable_name
        self._variable_type = _variable_type
        self._variable_type_qual = _variable_type_qual
        self._expression = _expression

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend([
            "_variable_name", "_variable_type", "_variable_type_qual", "_expression"
        ])
        return attrs

    # TODO Jason: Implement the functionality of this class


class ExpressionStatement(Statement):
    """Expression Statement"""
    # TODO Jason: Add docstring
    _left: Optional[Identifier] = None
    _index: List[Expression] = []  # Avoid a Mutable Default Arg
    _right: Expression

    def __init__(self,
                 _index: List[Expression],
                 _right: Expression,
                 _left: Optional[Identifier] = None,
                 ) -> None:
        super().__init__()
        self._left = _left
        self._index = _index
        self._right = _right
    
    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_left", "_index", "_right"])
        return attrs

    # TODO Jason: Implement the functionality of this class


class ForAllStatement(Statement):
    """For Loop Node"""

    def __init__(self,
                 _index: Expression,
                 _body: List[Statement]
                 ) -> None:
        super().__init__()
        self._index = _index
        self._body = _body

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_index", "_body"])
        return attrs

    # TODO Jason: Implement the functionality of this class


class BranchStatement(Statement):
    """A Branch (Conditional) Statement Block Node.

    Args:
        _predicate (Expression): Condition to Be Evaluated
        _true_body (List[Statement]): Body of Statements Evaluated if True
        _false_body (List[Statement]): Body of Statements Evaluated if False

    """

    def __init__(self,
                 _predicate: Expression,
                 _true_body: List[Statement],
                 _false_body: List[Statement],
                 ) -> None:
        super().__init__()
        self._predicate = _predicate
        self._true_body = _true_body
        self._false_body = _false_body

    def visit_attrs(self) -> List[str]:
        attrs = super().visit_attrs()
        attrs.extend(["_predicate", "_true_body", "_false_body"])
        return attrs

    # TODO Jason: Implement the functionality of this class
