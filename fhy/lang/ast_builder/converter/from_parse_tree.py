"""

"""
from logging import Logger
from typing import List, Optional

from antlr4 import ParseTreeWalker, ParserRuleContext

from fhy.lang.ast import ASTNode
from fhy.lang.parser import FhYListener, FhYParser

from ..builder import ASTBuilder
from ...span import Span, _Span
from ....utils.logger import get_logger


_log: Logger = get_logger(__name__)


def getSourceInfo(ctx: ParserRuleContext) -> Span:
    """Retrieves Line and Column Information from a Context"""
    start = ctx.start
    stop = ctx.stop
    span = Span(
        line=_Span(start.line, stop.line),
        column=_Span(start.column, stop.column)
    )
    return span


class ParseTreeConverter(FhYListener):
    # TODO Jason: Add docstring
    _builder: ASTBuilder
    _ast: Optional[ASTNode]

    def __init__(self, log: Logger = _log) -> None:
        super().__init__()
        self._builder = ASTBuilder()
        self._ast = None
        self._log = log

    def enterModule(self, ctx: FhYParser.ModuleContext) -> None:
        self._log.debug("Enter")
        self._builder.add_module()

    def exitModule(self, ctx: FhYParser.ModuleContext) -> None:
        self._log.debug(" Exit")
        self._builder.close_module_building()
        self._ast: ASTNode = self._builder.ast

    def enterComponent(self, ctx: FhYParser.ComponentContext) -> None:
        self._log.debug("Enter")
        if ctx.function_declaration() is not None:
            raise NotImplementedError("Function Declarations are not yet Supported.")
        elif ctx.function_definition() is not None:
            pass
        else:
            raise NotImplementedError()

    def exitComponent(self, ctx: FhYParser.ComponentContext) -> None:
        self._log.debug(" Exit")
        if any([ctx.function_declaration(), ctx.function_definition()]):
            self._builder.close_component_building()

    def enterFunction_header(self, ctx: FhYParser.Function_headerContext):
        self._log.debug("Enter")
        function_keyword: str = ctx.FUNCTION_KEYWORD().getText()
        function_name: str = ctx.IDENTIFIER().getText()
        if function_keyword == "proc":
            self._builder.add_procedure(function_name)
        elif function_keyword == "op":
            self._builder.add_operation(function_name)
        else:
            raise NotImplementedError()

    # def exitFunction_header(self, ctx:FhYParser.Function_headerContext):
    #     ...

    def enterFunction_arg(self, ctx: FhYParser.Function_argContext):
        self._log.debug("Enter")
        if (arg_name := ctx.IDENTIFIER()) is None:
            span: Span = getSourceInfo(ctx.parentCtx)
            raise SyntaxError(f"Function Argument Not given an Identifier: {span}")
        self._builder.add_argument(arg_name.getText())

    def exitFunction_arg(self, ctx: FhYParser.Function_argContext):
        self._log.debug(" Exit")
        self._builder.close_argument_building()

    def enterFunction_body(self, ctx: FhYParser.Function_bodyContext):
        self._log.debug("Enter")

    def exitFunction_body(self, ctx: FhYParser.Function_bodyContext):
        self._log.debug(" Exit")

    def enterAtom(self, ctx: FhYParser.AtomContext):
        self._log.debug(f"Enter")
        if ctx.literal() is not None:
            return
        self._builder.add_identifier(ctx.getText())

    # STATEMENT CONTEXTS
    def enterDeclaration_statement(self, ctx: FhYParser.Declaration_statementContext):
        self._log.debug("Enter")
        name = ctx.IDENTIFIER().getText()
        self._builder.open_declaration_statement(name)

    def exitDeclaration_statement(self, ctx: FhYParser.Declaration_statementContext):
        self._log.debug(" Exit")
        self._builder.close_declaration_statement()

    # def enterExpression_statement(self, ctx:FhYParser.Expression_statementContext):
    #     pass

    # def exitExpression_statement(self, ctx:FhYParser.Expression_statementContext):
    #     pass

    # def enterSelection_statement(self, ctx:FhYParser.Selection_statementContext):
    #     # We have two Statement Children of a Selection Statement, which
    #     # may or may not contain children themselves.
    #     self._builder.open_branch_statement()

    # def exitSelection_statement(self, ctx:FhYParser.Selection_statementContext):
    #     self._builder.close_branch_statement()

    # def enterIteration_statement(self, ctx:FhYParser.Iteration_statementContext):
    #     pass

    # def exitIteration_statement(self, ctx:FhYParser.Iteration_statementContext):
    #     pass

    def enterReturn_statement(self, ctx: FhYParser.Return_statementContext):
        self._log.debug("Enter")
        self._builder.open_return_statement()

    def exitReturn_statement(self, ctx: FhYParser.Return_statementContext):
        self._log.debug(" Exit")
        self._builder.close_return_statement()

    # TYPE CONTEXTS
    def enterQualified_type(self, ctx: FhYParser.Qualified_typeContext):
        self._log.debug("Enter")
        type_qualifier_name: str = ctx.IDENTIFIER().getText()
        self._builder.add_qualified_type(type_qualifier_name)

    def exitQualified_type(self, ctx: FhYParser.Qualified_typeContext):
        self._log.debug(" Exit")
        self._builder.close_qualified_type_building()

    # def enterType(self, ctx:FhYParser.TypeContext):
    #     pass

    # def exitType(self, ctx:FhYParser.TypeContext):
    #     pass

    # def enterTuple_type(self, ctx:FhYParser.Tuple_typeContext):
    #     pass

    # def exitTuple_type(self, ctx:FhYParser.Tuple_typeContext):
    #     pass

    def enterNumerical_type(self, ctx: FhYParser.Numerical_typeContext):
        self._log.debug("Enter")
        self._builder.add_numerical_type()

    # def exitNumerical_type(self, ctx:FhYParser.Numerical_typeContext):
    #     pass

    def enterDtype(self, ctx: FhYParser.DtypeContext):
        self._log.debug("Enter")
        numerical_type_name: str = ctx.IDENTIFIER().getText()
        self._builder.add_dtype(numerical_type_name)

    # def exitDtype(self, ctx:FhYParser.DtypeContext):
    #     pass

    def enterShape(self, ctx: FhYParser.ShapeContext):
        self._log.debug("Enter")
        self._builder.open_shape()

    def exitShape(self, ctx:FhYParser.ShapeContext):
        self._log.debug(" Exit")
        self._builder.close_shape()

    def enterIndex_type(self, ctx: FhYParser.Index_typeContext):
        self._log.debug("Enter")
        self._builder.open_index_type()

    def exitIndex_type(self, ctx:FhYParser.Index_typeContext):
        self._log.debug(" Exit")
        self._builder.close_index_type()

    def exitType(self, ctx: FhYParser.TypeContext):
        self._log.debug(" Exit")
        self._builder.close_type_building()

    # EXPRESSION CONTEXTS
    def enterExpression(self, ctx: FhYParser.ExpressionContext):
        self._log.debug("Enter")
        if ctx.primary_expression() is not None:
            ...

        elif ctx.nested_expression is not None:
            # We pass since a nested expression contains a child expression.
            # which will return back here anyways and be solved then.
            ...

        elif (unary := ctx.unary_expression) is not None:
            self._builder.add_unary_expression(unary.text)

        elif ctx.multiplicative_expression is not None:
            op = ctx.DIVISION() or ctx.MULTIPLICATION()
            self._builder.add_binary_expression(op.getText())

        elif ctx.additive_expression is not None:
            op = ctx.ADDITION() or ctx.SUBTRACTION()
            self._builder.add_binary_expression(op.getText())

        elif ctx.shift_expression is not None:
            op = ctx.LEFT_SHIFT() or ctx.RIGHT_SHIFT()
            self._builder.add_binary_expression(op.getText())

        elif ctx.relational_expression is not None:
            op = (
                ctx.LESS_THAN()
                or ctx.LESS_THAN_OR_EQUAL()
                or ctx.GREATER_THAN()
                or ctx.GREATER_THAN_OR_EQUAL()
            )
            self._builder.add_binary_expression(op.getText())

        elif ctx.equality_expression is not None:
            op = ctx.EQUAL_TO() or ctx.NOT_EQUAL_TO()
            self._builder.add_binary_expression(op.getText())

        elif ctx.and_expression is not None:
            self._builder.add_binary_expression(ctx.AND().getText())

        elif ctx.exclusive_or_expression is not None:
            self._builder.add_binary_expression(ctx.EXCLUSIVE_OR().getText())

        elif ctx.or_expression is not None:
            self._builder.add_binary_expression(ctx.OR().getText())

        elif ctx.logical_and_expression is not None:
            self._builder.add_binary_expression(ctx.LOGICAL_AND().getText())

        elif ctx.logical_or_expression is not None:
            self._builder.add_binary_expression(ctx.LOGICAL_OR().getText())

        elif ctx.ternary_expression is not None:
            assert ctx.QUESTION_MARK().getText() == "?"
            self._builder.open_ternary_expression()

        else:
            raise NotImplementedError("Unknown Expression Not Implemented")

    def exitExpression(self, ctx: FhYParser.ExpressionContext):
        self._log.debug(" Exit")
        if ctx.primary_expression() is not None:
            ...

        elif ctx.nested_expression is not None:
            # We pass since a nested expression contains a child expression.
            # which will return back here anyways and be solved then.
            ...

        elif ctx.unary_expression is not None:
            self._builder.close_unary_expression()

        elif any(
            i is not None
            for i in (
                ctx.multiplicative_expression,
                ctx.additive_expression,
                ctx.shift_expression,
                ctx.relational_expression,
                ctx.equality_expression,
                ctx.and_expression,
                ctx.or_expression,
                ctx.logical_and_expression,
                ctx.logical_or_expression,
            )
        ):
            self._builder.close_binary_expression()

        elif ctx.ternary_expression is not None:
            self._builder.close_ternary_expression()

        else:
            raise NotImplementedError("Unknown Expression Not Implemented")

    def enterPrimary_expression(self, ctx: FhYParser.Primary_expressionContext):
        self._log.debug("Enter")
        if ctx.tuple_access_expression is not None:
            ...

        elif ctx.function_expression is not None:
            ...

        elif ctx.tensor_access_expression is not None:
            ...

        elif ctx.atom() is not None:
            ...

        else:
            raise NotImplementedError("Unknown Primary Expression")

    def exitStatement(self, ctx: FhYParser.StatementContext):
        self._log.debug(" Exit")
        self._builder.close_statement()

    # LITERALS
    def enterLiteral(self, ctx: FhYParser.LiteralContext):
        self._log.debug("Enter")
        # TODO: Capturing Floats is Not Working Here...
        if (floats := ctx.float_literal()) is not None:
            value = float(floats.getText())

        elif (integer := ctx.INT_LITERAL()) is not None:
            value = int(integer.getText())

        # TODO: Handle Complex Values
        else:
            raise NotImplementedError("Unknown Type Literal")
        self._builder.add_literal(value)

    @property
    def ast(self) -> Optional[ASTNode]:
        return self._ast


def from_parse_tree(parse_tree: FhYParser.ModuleContext) -> ASTNode:
    # TODO Jason: Add docstring
    converter = ParseTreeConverter()
    walker = ParseTreeWalker()
    walker.walk(converter, parse_tree)
    assert len(converter._builder._node_stack) == 0, "Incomplete AST Build."
    if converter.ast is None:
        # TODO Jason: Implement a better error for this ast conversion failure
        raise Exception()

    return converter.ast
