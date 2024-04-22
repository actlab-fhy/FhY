""" """

from collections import ChainMap
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple, Union

from antlr4 import ParserRuleContext, ParseTreeWalker

from fhy import ir
from fhy.lang import ast
from fhy.lang.parser import FhYListener, FhYParser, FhYVisitor

from ....utils.logger import get_logger
from ...span import Span
from ..builder import ASTBuilder
from ..builder_frame import ASTBuilderFrame


def _get_source_info(ctx: ParserRuleContext) -> Span:
    """Retrieves line and column information from a context"""
    # start = ctx.start
    # stop = ctx.stop
    # return Span(start.line, stop.line, start.column, stop.column)
    return Span(0, 0, 0, 0)


def _initialize_builtin_identifiers() -> Dict[str, ir.Identifier]:
    return {
        "sum": ir.Identifier("sum"),
    }


# TODO: Change function signatures to enable type checking
# TODO: after the todo abovce remove all the unnecessary asserts for types


class ParseTreeConverter(FhYVisitor):
    _scopes: ChainMap[str, ir.Identifier]

    def __init__(self) -> None:
        self._scopes = ChainMap(_initialize_builtin_identifiers())

    def _open_scope(self) -> None:
        self._scopes = self._scopes.new_child()

    def _close_scope(self) -> None:
        self._scopes = self._scopes.parents

    def _get_identifier(self, name_hint: str) -> ir.Identifier:
        if name_hint in self._scopes:
            return self._scopes[name_hint]
        else:
            identifier = ir.Identifier(name_hint)
            self._scopes[name_hint] = identifier
            return identifier

    def convert(self, ctx: FhYParser.ModuleContext) -> ast.Module:
        node = self.visitModule(ctx)
        assert isinstance(node, ast.Module), f'Expected "Module", got {type(node)}'
        return node

    # =====================
    # MODULE VISITORS
    # =====================
    def visitModule(self, ctx: FhYParser.ModuleContext) -> Any:
        self._open_scope()
        components: List[ast.Component] = []
        for component_ctx in ctx.component():
            component = self.visitComponent(component_ctx)
            assert isinstance(
                component, ast.Component
            ), f'Expected "Component", got {type(component)}'
            components.append(component)
        span = _get_source_info(ctx)
        self._close_scope()
        return ast.Module(components=components, span=span)

    # =====================
    # FUNCTION VISITORS
    # =====================
    def visitFunction_declaration(
        self, ctx: FhYParser.Function_declarationContext
    ) -> Any:
        # TODO: Implement
        raise NotImplementedError()

    def visitFunction_definition(
        self, ctx: FhYParser.Function_definitionContext
    ) -> Any:
        self._open_scope()
        # TODO: add template types and indices (3rd and 4th returned values here)
        keyword, name, _, _, args, return_type = self.visitFunction_header(
            ctx.function_header()
        )

        body_ctx = ctx.function_body()
        body = self.visitFunction_body(body_ctx)

        span = _get_source_info(ctx)

        self._close_scope()

        if keyword == "proc":
            if return_type is not None:
                raise Exception()

            return ast.Procedure(name=name, args=args, body=body, span=span)
        elif keyword == "op":
            return ast.Operation(
                name=name, args=args, return_type=return_type, body=body, span=span
            )
        else:
            raise NotImplementedError()

    def visitFunction_header(self, ctx: FhYParser.Function_headerContext) -> Any:
        keyword: str = ctx.FUNCTION_KEYWORD().getText()

        name_hint: str = ctx.IDENTIFIER().getText()
        name = self._get_identifier(name_hint)

        args_ctx: FhYParser.Function_argsContext = ctx.function_args(0)
        args: List[ast.Argument] = self.visitFunction_args(args_ctx)

        return_type: Optional[ast.QualifiedType] = None
        if (return_type_ctx := ctx.qualified_type()) is not None:
            return_type = self.visitQualified_type(return_type_ctx)

        return keyword, name, None, None, args, return_type

    def visitFunction_args(self, ctx: FhYParser.Function_argsContext) -> Any:
        args: List[ast.Argument] = []
        if ctx.function_arg() is not None:
            for arg_ctx in ctx.function_arg():
                args.append(self.visitFunction_arg(arg_ctx))
        return args

    def visitFunction_arg(self, ctx: FhYParser.Function_argContext) -> Any:
        qualified_type = self.visitQualified_type(ctx.qualified_type())
        name_hint: str = ctx.IDENTIFIER().getText()
        name = self._get_identifier(name_hint)

        span = _get_source_info(ctx)
        return ast.Argument(qualified_type=qualified_type, name=name, span=span)

    def visitFunction_body(self, ctx: FhYParser.Function_bodyContext) -> Any:
        return self.visitStatement_series(ctx.statement_series())

    # =====================
    # STATEMENT VISITORS
    # =====================
    def visitStatement_series(self, ctx: FhYParser.Statement_seriesContext):
        statements: List[ast.Statement] = []
        if ctx.statement() is not None:
            for statement_ctx in ctx.statement():
                statement = self.visitStatement(statement_ctx)
                assert isinstance(
                    statement, ast.Statement
                ), f'Expected "Statement", got {type(statement)}'
                statements.append(statement)
        return statements

    def visitDeclaration_statement(
        self, ctx: FhYParser.Declaration_statementContext
    ) -> Any:
        qualified_type = self.visitQualified_type(ctx.qualified_type())
        name_hint: str = ctx.IDENTIFIER().getText()
        name = self._get_identifier(name_hint)
        expression = None
        if (expression_ctx := ctx.expression()) is not None:
            expression = self.visitExpression(expression_ctx)
        span = _get_source_info(ctx)
        return ast.DeclarationStatement(
            variable_type=qualified_type,
            variable_name=name,
            expression=expression,
            span=span,
        )

    def visitExpression_statement(
        self, ctx: FhYParser.Expression_statementContext
    ) -> Any:
        left_expression = None
        if (primitive_expression_ctx := ctx.primitive_expression()) is not None:
            left_expression = self.visitPrimitive_expression(primitive_expression_ctx)
            assert isinstance(
                left_expression, ast.Expression
            ), f'Expected "Expression", got {type(left_expression)}'

        right_expression_ctx = ctx.expression()
        right_expression = self.visitExpression(right_expression_ctx)
        assert isinstance(
            right_expression, ast.Expression
        ), f'Expected "Expression", got {type(right_expression)}'

        span = _get_source_info(ctx)

        return ast.ExpressionStatement(
            left=left_expression, right=right_expression, span=span
        )

    def visitSelection_statement(self, ctx: FhYParser.Selection_statementContext):
        condition_ctx = ctx.expression()
        condition = self.visitExpression(condition_ctx)
        assert isinstance(
            condition, ast.Expression
        ), f'Expected "Expression", got {type(condition)}'
        self._open_scope()
        true_body_ctx = ctx.statement_series(0)
        true_body = self.visitStatement_series(true_body_ctx)
        self._close_scope()
        false_body = []
        if (false_body_ctx := ctx.statement_series(1)) is not None:
            self._open_scope()
            false_body = self.visitStatement_series(false_body_ctx)
            self._close_scope()
        span = _get_source_info(ctx)
        return ast.SelectionStatement(
            condition=condition, true_body=true_body, false_body=false_body, span=span
        )

    def visitIteration_statement(self, ctx: FhYParser.Iteration_statementContext):
        index_ctx = ctx.expression()
        index = self.visitExpression(index_ctx)
        assert isinstance(
            index, ast.Expression
        ), f'Expected "Expression", got {type(index)}'
        self._open_scope()
        body_ctx = ctx.statement_series()
        body = self.visitStatement_series(body_ctx)
        assert all(
            isinstance(statement, ast.Statement) for statement in body
        ), 'Expected all elements to be "Statement"'
        self._close_scope()
        span = _get_source_info(ctx)
        return ast.ForAllStatement(index=index, body=body, span=span)

    def visitReturn_statement(self, ctx: FhYParser.Return_statementContext) -> Any:
        expression_ctx = ctx.expression()
        expression = self.visitExpression(expression_ctx)
        assert isinstance(
            expression, ast.Expression
        ), f'Expected "Expression", got {type(expression)}'
        span = _get_source_info(ctx)
        return ast.ReturnStatement(expression=expression, span=span)

    # =====================
    # EXPRESSION VISITORS
    # =====================
    def visitExpression_list(self, ctx: FhYParser.Expression_listContext) -> Any:
        expressions: List[ast.Expression] = []
        if ctx.expression() is not None:
            for expression_ctx in ctx.expression():
                expressions.append(self.visitExpression(expression_ctx))
        return expressions

    def visitExpression(self, ctx: FhYParser.ExpressionContext) -> Any:
        span = _get_source_info(ctx)
        if ctx.nested_expression is not None:
            return self.visitExpression(ctx.expression(0))

        elif ctx.unary_expression is not None:
            operand = self.visitExpression(ctx.expression(0))
            assert isinstance(
                operand, ast.Expression
            ), f'Expected "Expression", got {type(operand)}'
            operator_ctx = ctx.SUBTRACTION() or ctx.BITWISE_NOT() or ctx.LOGICAL_NOT()
            assert operator_ctx is not None, "Expected unary operator"
            return ast.UnaryExpression(
                span=span,
                operation=ast.UnaryOperation(operator_ctx.getText()),
                expression=operand,
            )

        elif any(
            [
                ctx.multiplicative_expression,
                ctx.additive_expression,
                ctx.shift_expression,
                ctx.relational_expression,
                ctx.equality_expression,
                ctx.and_expression,
                ctx.or_expression,
                ctx.logical_and_expression,
                ctx.logical_or_expression,
            ]
        ):
            left = self.visitExpression(ctx.expression(0))
            assert isinstance(
                left, ast.Expression
            ), f'Expected "Expression", got {type(left)}'
            right = self.visitExpression(ctx.expression(1))
            assert isinstance(
                right, ast.Expression
            ), f'Expected "Expression", got {type(right)}'
            operator = (
                ctx.MULTIPLICATION()
                or ctx.DIVISION()
                or ctx.ADDITION()
                or ctx.SUBTRACTION()
                or ctx.LEFT_SHIFT()
                or ctx.RIGHT_SHIFT()
                or ctx.LESS_THAN()
                or ctx.LESS_THAN_OR_EQUAL()
                or ctx.GREATER_THAN()
                or ctx.GREATER_THAN_OR_EQUAL()
                or ctx.EQUAL_TO()
                or ctx.NOT_EQUAL_TO()
                or ctx.AND()
                or ctx.OR()
                or ctx.LOGICAL_AND()
                or ctx.LOGICAL_OR()
            )
            assert operator is not None, "Expected binary operator"
            return ast.BinaryExpression(
                span=span,
                operation=ast.BinaryOperation(operator.getText()),
                left=left,
                right=right,
            )

        elif ctx.ternary_expression is not None:
            condition = self.visitExpression(ctx.expression(0))
            assert isinstance(
                condition, ast.Expression
            ), f'Expected "Expression", got {type(condition)}'
            true_expression = self.visitExpression(ctx.expression(1))
            assert isinstance(
                true_expression, ast.Expression
            ), f'Expected "Expression", got {type(true_expression)}'
            false_expression = self.visitExpression(ctx.expression(2))
            assert isinstance(
                false_expression, ast.Expression
            ), f'Expected "Expression", got {type(false_expression)}'
            return ast.TernaryExpression(
                span=span,
                condition=condition,
                true=true_expression,
                false=false_expression,
            )

        elif (primitive_expression_ctx := ctx.primitive_expression()) is not None:
            primitive_expression = self.visitPrimitive_expression(
                primitive_expression_ctx
            )
            assert isinstance(
                primitive_expression, ast.Expression
            ), f'Expected "Expression", got {type(primitive_expression)}'
            return primitive_expression

        else:
            raise Exception()

    def visitPrimitive_expression(
        self, ctx: FhYParser.Primitive_expressionContext
    ) -> Any:
        span = _get_source_info(ctx)
        # if ctx.tuple_access_expression is not None:
        #     primitive_expression_ctx = ctx.primitive_expression()
        #     primitive_expression = self.visitPrimitive_expression(primitive_expression_ctx)
        #     assert isinstance(primitive_expression, ast.Expression), f"Expected \"Expression\", got {type(primitive_expression)}"
        #     return ast.TupleAccessExpression(span=span, tuple_expression=primitive_expression, element_index=int(ctx.INT_LITERAL().getText()))

        if ctx.function_expression is not None:
            function_expression_ctx = ctx.primitive_expression()
            function_expression = self.visitPrimitive_expression(
                function_expression_ctx
            )
            assert isinstance(
                function_expression, ast.Expression
            ), f'Expected "Expression", got {type(function_expression)}'

            expression_list_counter: int = 0

            template_types: List[ast.Expression] = []
            if ctx.LESS_THAN() is not None and ctx.GREATER_THAN() is not None:
                template_types = self.visitExpression_list(
                    ctx.expression_list(expression_list_counter)
                )
                assert all(
                    isinstance(expr, ast.Expression) for expr in template_types
                ), 'Expected all elements to be "Expression"'
                expression_list_counter += 1

            indices: List[ast.Expression] = []
            if ctx.OPEN_BRACKET() is not None and ctx.CLOSE_BRACKET() is not None:
                indices = self.visitExpression_list(
                    ctx.expression_list(expression_list_counter)
                )
                assert all(
                    isinstance(expr, ast.Expression) for expr in indices
                ), 'Expected all elements to be "Expression"'
                expression_list_counter += 1

            args = self.visitExpression_list(
                ctx.expression_list(expression_list_counter)
            )
            assert all(
                isinstance(expr, ast.Expression) for expr in args
            ), 'Expected all elements to be "Expression"'

            return ast.FunctionExpression(
                function=function_expression,
                template_types=template_types,
                indices=indices,
                args=args,
                span=span,
            )

        elif ctx.array_access_expression is not None:
            array_expression_ctx = ctx.primitive_expression()
            array_expression = self.visitPrimitive_expression(array_expression_ctx)
            assert isinstance(
                array_expression, ast.Expression
            ), f'Expected "Expression", got {type(array_expression)}'

            indices_ctx = ctx.expression_list(0)
            indices = self.visitExpression_list(indices_ctx)
            assert all(
                isinstance(expr, ast.Expression) for expr in indices
            ), 'Expected all elements to be "Expression"'

            return ast.ArrayAccessExpression(
                array_expression=array_expression, indices=indices, span=span
            )

        elif (atom_ctx := ctx.atom()) is not None:
            atom_expression = self.visitAtom(atom_ctx)
            assert isinstance(
                atom_expression, ast.Expression
            ), f'Expected "Expression", got {type(atom_expression)}'
            return atom_expression

        else:
            raise Exception()

    def visitAtom(self, ctx: FhYParser.AtomContext) -> Any:
        # TODO: add tuples
        span = _get_source_info(ctx)
        if (identifier_ctx := ctx.IDENTIFIER()) is not None:
            identifer = self._get_identifier(identifier_ctx.getText())
            return ast.IdentifierExpression(identifier=identifer, span=span)

        elif (literal_ctx := ctx.literal()) is not None:
            return self.visitLiteral(literal_ctx)

        else:
            raise NotImplementedError()

    def visitLiteral(self, ctx: FhYParser.LiteralContext):
        if (int_literal_ctx := ctx.INT_LITERAL()) is not None:
            int_literal_str: str = int_literal_ctx.getText()

            if int_literal_str.startswith(("0x", "0X")):
                base = 16
            elif int_literal_str.startswith(("0b", "0B")):
                base = 2
            elif int_literal_str.startswith(("0o", "0O")):
                base = 8
            else:
                base = 10

            return ast.IntLiteral(span=None, value=int(int_literal_str, base=base))

        elif (float_literal_ctx := ctx.FLOAT_LITERAL()) is not None:
            float_literal = ast.FloatLiteral(
                span=None, value=float(float_literal_ctx.getText())
            )
            return float_literal

        else:
            raise Exception()

    # =====================
    # TYPE VISITORS
    # =====================
    def visitQualified_type(self, ctx: FhYParser.Qualified_typeContext) -> Any:
        type_qualifier: Optional[ir.TypeQualifier] = None
        if (type_qualifier_ctx := ctx.IDENTIFIER()) is not None:
            type_qualifier = ir.TypeQualifier(type_qualifier_ctx.getText())
        base_type = self.visitType(ctx.type_())
        span = _get_source_info(ctx)
        return ast.QualifiedType(
            base_type=base_type, type_qualifier=type_qualifier, span=span
        )

    def visitNumerical_type(self, ctx: FhYParser.Numerical_typeContext) -> Any:
        data_type = self.visitDtype(ctx.dtype())
        assert isinstance(
            data_type, ir.DataType
        ), f'Expected "DataType", got {type(data_type)}'
        shape: List[ast.Expression] = []
        if (shape_ctx := ctx.expression_list()) is not None:
            shape = self.visitExpression_list(shape_ctx)
            assert all(
                isinstance(expr, ast.Expression) for expr in shape
            ), 'Expected all elements to be "Expression"'
        return ir.NumericalType(data_type=data_type, shape=shape)

    def visitDtype(self, ctx: FhYParser.DtypeContext) -> Any:
        return ir.DataType(ir.PrimitiveDataType(ctx.IDENTIFIER().getText()))

    def visitIndex_type(self, ctx: FhYParser.Index_typeContext) -> Any:
        low, high, stride = self.visitRange(ctx.range_())
        assert isinstance(
            low, ast.Expression
        ), f'Expected "Expression", got {type(low)}'
        assert isinstance(
            high, ast.Expression
        ), f'Expected "Expression", got {type(high)}'
        if stride is not None:
            assert isinstance(
                stride, ast.Expression
            ), f'Expected "Expression", got {type(stride)}'
        return ir.IndexType(lower_bound=low, upper_bound=high, stride=stride)

    def visitRange(self, ctx: FhYParser.RangeContext) -> Any:
        low_ctx = ctx.expression(0)
        low = self.visitExpression(low_ctx)
        assert isinstance(
            low, ast.Expression
        ), f'Expected "Expression", got {type(low)}'

        high_ctx = ctx.expression(1)
        high = self.visitExpression(high_ctx)
        assert isinstance(
            high, ast.Expression
        ), f'Expected "Expression", got {type(high)}'

        stride = None
        if (stride_ctx := ctx.expression(2)) is not None:
            stride = self.visitExpression(stride_ctx)
            assert isinstance(
                stride, ast.Expression
            ), f'Expected "Expression", got {type(stride)}'

        return low, high, stride

    def visitTuple_type(self, ctx: FhYParser.Tuple_typeContext) -> ir.type.TupleType:
        types: List[ir.Type] = []
        if ctx.OPEN_PARENTHESES() is None or ctx.CLOSE_PARENTHESES() is None:
            raise SyntaxError("Invalid Tuple Expression")
        if (context := ctx.type_()) is not None:
            for t in context:
                types.append(self.visitType(t))

        return ir.type.TupleType(types=types)


def from_parse_tree(parse_tree: FhYParser.ModuleContext) -> ast.ASTNode:
    # TODO Jason: Add docstring
    converter = ParseTreeConverter()
    _ast = converter.visitModule(parse_tree)
    if _ast is None:
        raise Exception()

    return _ast
