""" """

from logging import Logger
from typing import Any, List, Tuple, Optional, Union

from antlr4 import ParserRuleContext, ParseTreeWalker

from fhy import ir
from fhy.lang import ast
from ..builder_frame import ASTBuilderFrame
from fhy.lang.parser import FhYListener, FhYParser, FhYVisitor

from ....utils.logger import get_logger
from ...span import Span
from ..builder import ASTBuilder

_log: Logger = get_logger(__name__)


# TODO: Consider making the builders immutable so they need to be popped, updated, and pushed again rather than modified in-place


def _get_source_info(ctx: ParserRuleContext) -> Span:
    """Retrieves line and column information from a context"""
    # start = ctx.start
    # stop = ctx.stop
    # return Span(start.line, stop.line, start.column, stop.column)
    return Span(0, 0, 0, 0)


class ParseTreeConverter(FhYVisitor):
    _builder: ASTBuilder

    def __init__(self) -> None:
        self._builder = ASTBuilder()

    def convert(self, ctx: FhYParser.ModuleContext) -> ast.Module:
        node = self.visitModule(ctx)
        assert isinstance(node, ast.Module), f"Expected \"Module\", got {type(node)}"
        return node

    # =====================
    # MODULE VISITORS
    # =====================
    def visitModule(self, ctx: FhYParser.ModuleContext) -> Any:
        components: List[ast.Component] = []
        for component_ctx in ctx.component():
            component = self.visitComponent(component_ctx)
            assert isinstance(component, ast.Component), f"Expected \"Component\", got {type(component)}"
            components.append(component)
        span = _get_source_info(ctx)
        return ast.Module(components=components, span=span)

    # =====================
    # FUNCTION VISITORS
    # =====================
    def visitFunction_declaration(
        self,
        ctx: FhYParser.Function_declarationContext
    ) -> Any:
        # TODO: Implement
        raise NotImplementedError()

    def visitFunction_definition(
        self,
        ctx: FhYParser.Function_definitionContext
    ) -> Any:
        # TODO: add template types and indices (3rd and 4th returned values here)
        keyword, name, _, _, args, return_type = self.visitFunction_header(
            ctx.function_header()
        )

        body_ctx = ctx.function_body()
        body = self.visitFunction_body(body_ctx)

        span = _get_source_info(ctx)

        if keyword == "proc":
            if return_type is not None:
                raise Exception()

            return ast.Procedure(
                name=name, args=args, body=body, span=span
            )
        elif keyword == "op":
            return ast.Operation(
                name=name, args=args, return_type=return_type, body=body,
                span=span
            )
        else:
            raise NotImplementedError()

    def visitFunction_header(
        self,
        ctx: FhYParser.Function_headerContext
    ) -> Any:
        keyword: str = ctx.FUNCTION_KEYWORD().getText()

        name: ir.Identifier = ir.Identifier(ctx.IDENTIFIER().getText())

        args_ctx: FhYParser.Function_argsContext = ctx.function_args(0)
        args: List[ast.Argument] = self.visitFunction_args(args_ctx)

        return_type: Optional[ast.QualifiedType] = None
        if (return_type_ctx := ctx.qualified_type()) is not None:
            return_type = self.visitQualified_type(return_type_ctx)

        return keyword, name, None, None, args, return_type

    def visitFunction_args(
        self,
        ctx: FhYParser.Function_argsContext
    ) -> Any:
        args: List[ast.Argument] = []
        if ctx.function_arg() is not None:
            for arg_ctx in ctx.function_arg():
                args.append(self.visitFunction_arg(arg_ctx))
        return args

    def visitFunction_arg(
        self,
        ctx: FhYParser.Function_argContext
    ) -> Any:
        qualified_type = self.visitQualified_type(ctx.qualified_type())
        name = ir.Identifier(ctx.IDENTIFIER().getText())
        span = _get_source_info(ctx)
        return ast.Argument(qualified_type=qualified_type, name=name, span=span)

    def visitFunction_body(
        self,
        ctx: FhYParser.Function_bodyContext
    ) -> Any:
        statements: List[ast.Statement] = []
        if ctx.statement() is not None:
            for statement_ctx in ctx.statement():
                statement = self.visitStatement(statement_ctx)
                assert isinstance(statement, ast.Statement), f"Expected \"Statement\", got {type(statement)}"
                statements.append(statement)
        return statements

    # =====================
    # STATEMENT VISITORS
    # =====================
    def visitDeclaration_statement(self, ctx: FhYParser.Declaration_statementContext) -> Any:
        qualified_type = self.visitQualified_type(ctx.qualified_type())
        name = ir.Identifier(ctx.IDENTIFIER().getText())
        expression = None
        if (expression_ctx := ctx.expression()) is not None:
            expression = self.visitExpression(expression_ctx)
        span = _get_source_info(ctx)
        return ast.DeclarationStatement(variable_type=qualified_type, variable_name=name, expression=expression, span=span)

    def visitExpression_statement(self, ctx: FhYParser.Expression_statementContext) -> Any:
        left_expression = None
        if (primitive_expression_ctx := ctx.primitive_expression()) is not None:
            left_expression = self.visitPrimitive_expression(primitive_expression_ctx)
            assert isinstance(left_expression, ast.Expression), f"Expected \"Expression\", got {type(left_expression)}"

        right_expression_ctx = ctx.expression()
        right_expression = self.visitExpression(right_expression_ctx)
        assert isinstance(right_expression, ast.Expression), f"Expected \"Expression\", got {type(right_expression)}"

        span = _get_source_info(ctx)

        return ast.ExpressionStatement(left=left_expression, right=right_expression, span=span)

    def visitReturn_statement(self, ctx: FhYParser.Return_statementContext) -> Any:
        expression_ctx = ctx.expression()
        expression = self.visitExpression(expression_ctx)
        assert isinstance(expression, ast.Expression), f"Expected \"Expression\", got {type(expression)}"
        span = _get_source_info(ctx)
        return ast.ReturnStatement(expression=expression, span=span)

    # =====================
    # EXPRESSION VISITORS
    # =====================
    def visitExpression_list(
        self,
        ctx: FhYParser.Expression_listContext
    ) -> Any:
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
            assert isinstance(operand, ast.Expression), f"Expected \"Expression\", got {type(operand)}"
            operator_ctx = ctx.SUBTRACTION() or ctx.BITWISE_NOT() or ctx.LOGICAL_NOT()
            assert operator_ctx is not None, "Expected unary operator"
            return ast.UnaryExpression(span=span, operation=ast.UnaryOperation(operator_ctx.getText()), expression=operand)

        elif any([ctx.multiplicative_expression, ctx.additive_expression, ctx.shift_expression, ctx.relational_expression, ctx.equality_expression, ctx.and_expression, ctx.or_expression, ctx.logical_and_expression, ctx.logical_or_expression]):
            left = self.visitExpression(ctx.expression(0))
            assert isinstance(left, ast.Expression), f"Expected \"Expression\", got {type(left)}"
            right = self.visitExpression(ctx.expression(1))
            assert isinstance(right, ast.Expression), f"Expected \"Expression\", got {type(right)}"
            operator = ctx.MULTIPLICATION() or ctx.DIVISION() or ctx.ADDITION() or ctx.SUBTRACTION() or ctx.LEFT_SHIFT() or ctx.RIGHT_SHIFT() or ctx.LESS_THAN() or ctx.LESS_THAN_OR_EQUAL() or ctx.GREATER_THAN() or ctx.GREATER_THAN_OR_EQUAL() or ctx.EQUAL_TO() or ctx.NOT_EQUAL_TO() or ctx.AND() or ctx.OR() or ctx.LOGICAL_AND() or ctx.LOGICAL_OR()
            assert operator is not None, "Expected binary operator"
            return ast.BinaryExpression(span=span, operation=ast.BinaryOperation(operator.getText()), left=left, right=right)

        elif ctx.ternary_expression is not None:
            condition = self.visitExpression(ctx.expression(0))
            assert isinstance(condition, ast.Expression), f"Expected \"Expression\", got {type(condition)}"
            true_expression = self.visitExpression(ctx.expression(1))
            assert isinstance(true_expression, ast.Expression), f"Expected \"Expression\", got {type(true_expression)}"
            false_expression = self.visitExpression(ctx.expression(2))
            assert isinstance(false_expression, ast.Expression), f"Expected \"Expression\", got {type(false_expression)}"
            return ast.TernaryExpression(span=span, condition=condition, true=true_expression, false=false_expression)

        elif (primitive_expression_ctx := ctx.primitive_expression()) is not None:
            primitive_expression = self.visitPrimitive_expression(primitive_expression_ctx)
            assert isinstance(primitive_expression, ast.Expression), f"Expected \"Expression\", got {type(primitive_expression)}"
            return primitive_expression

        else:
            raise Exception()

    def visitPrimitive_expression(self, ctx: FhYParser.Primitive_expressionContext) -> Any:
        span = _get_source_info(ctx)
        # if ctx.tuple_access_expression is not None:
        #     primitive_expression_ctx = ctx.primitive_expression()
        #     primitive_expression = self.visitPrimitive_expression(primitive_expression_ctx)
        #     assert isinstance(primitive_expression, ast.Expression), f"Expected \"Expression\", got {type(primitive_expression)}"
        #     return ast.TupleAccessExpression(span=span, tuple_expression=primitive_expression, element_index=int(ctx.INT_LITERAL().getText()))

        if ctx.function_expression is not None:
            function_expression_ctx = ctx.primitive_expression()
            function_expression = self.visitPrimitive_expression(function_expression_ctx)
            assert isinstance(function_expression, ast.Expression), f"Expected \"Expression\", got {type(function_expression)}"

            expression_list_counter: int = 0

            template_types: List[ast.Expression] = []
            if ctx.LESS_THAN() is not None and ctx.GREATER_THAN() is not None:
                template_types = self.visitExpression_list(ctx.expression_list(expression_list_counter))
                assert all(isinstance(expr, ast.Expression) for expr in template_types), "Expected all elements to be \"Expression\""
                expression_list_counter += 1

            indices: List[ast.Expression] = []
            if ctx.OPEN_BRACKET() is not None and ctx.CLOSE_BRACKET() is not None:
                indices = self.visitExpression_list(ctx.expression_list(expression_list_counter))
                assert all(isinstance(expr, ast.Expression) for expr in indices), "Expected all elements to be \"Expression\""
                expression_list_counter += 1

            args = self.visitExpression_list(ctx.expression_list(expression_list_counter))
            assert all(isinstance(expr, ast.Expression) for expr in args), "Expected all elements to be \"Expression\""

            return ast.FunctionExpression(function=function_expression, template_types=template_types, indices=indices, args=args, span=span)

        elif ctx.array_access_expression is not None:
            array_expression_ctx = ctx.primitive_expression()
            array_expression = self.visitPrimitive_expression(array_expression_ctx)
            assert isinstance(array_expression, ast.Expression), f"Expected \"Expression\", got {type(array_expression)}"

            indices_ctx = ctx.expression_list(0)
            indices = self.visitExpression_list(indices_ctx)
            assert all(isinstance(expr, ast.Expression) for expr in indices), "Expected all elements to be \"Expression\""

            return ast.ArrayAccessExpression(array_expression=array_expression, indices=indices, span=span)

        elif (atom_ctx := ctx.atom()) is not None:
            atom_expression = self.visitAtom(atom_ctx)
            assert isinstance(atom_expression, ast.Expression), f"Expected \"Expression\", got {type(atom_expression)}"
            return atom_expression

        else:
            raise Exception()

    def visitAtom(self, ctx: FhYParser.AtomContext) -> Any:
        # TODO: add tuples
        span = _get_source_info(ctx)
        if (identifier_ctx := ctx.IDENTIFIER()) is not None:
            identifer = ir.Identifier(identifier_ctx.getText())
            return ast.IdentifierExpression(identifier=identifer, span=span)

        elif (literal_ctx := ctx.literal()) is not None:
            return self.visitLiteral(literal_ctx)

        else:
            raise NotImplementedError()

    def visitLiteral(self, ctx: FhYParser.LiteralContext):
        if (int_literal_ctx := ctx.INT_LITERAL()) is not None:
            return ast.IntLiteral(span=None, value=int(int_literal_ctx.getText()))

        elif (float_literal_ctx := ctx.float_literal()) is not None:
            float_literal = self.visitFloat_literal(float_literal_ctx)
            assert isinstance(float_literal, ast.FloatLiteral), f"Expected \"FloatLiteral\", got {type(float_literal)}"
            return float_literal

        else:
            raise Exception()


    def visitDecimal_float_literal(self, ctx: FhYParser.Decimal_float_literalContext) -> Any:
        span = _get_source_info(ctx)
        if (fraction_part_ctx := ctx.fraction_part()) is not None:
            literal_piece = self.visitFraction_part(fraction_part_ctx)
            assert isinstance(literal_piece, float), f"Expected \"float\", got {type(literal_piece)}"
            if (exponent_part_ctx := ctx.EXPONENT_PART()) is not None:
                return ast.FloatLiteral(span=span, value=literal_piece * 10 ** int(exponent_part_ctx.getText()))
            else:
                return ast.FloatLiteral(span=span, value=literal_piece)

        else:
            return ast.FloatLiteral(span=span, value=float(ctx.getText()))


    def visitFraction_part(self, ctx: FhYParser.Fraction_partContext) -> Any:
        return float(ctx.getText())


    # =====================
    # TYPE VISITORS
    # =====================
    def visitQualified_type(
        self,
        ctx: FhYParser.Qualified_typeContext
    ) -> Any:
        type_qualifier: Optional[ir.TypeQualifier] = None
        if (type_qualifier_ctx := ctx.IDENTIFIER()) is not None:
            type_qualifier = ir.TypeQualifier(type_qualifier_ctx.getText())
        base_type = self.visitType(ctx.type_())
        span = _get_source_info(ctx)
        return ast.QualifiedType(base_type=base_type,
                                 type_qualifier=type_qualifier, span=span)

    def visitNumerical_type(self, ctx: FhYParser.Numerical_typeContext) -> Any:
        data_type = self.visitDtype(ctx.dtype())
        assert isinstance(data_type, ir.DataType), f"Expected \"DataType\", got {type(data_type)}"
        shape: List[ast.Expression] = []
        if (shape_ctx := ctx.expression_list()) is not None:
            shape = self.visitExpression_list(shape_ctx)
            assert all(isinstance(expr, ast.Expression) for expr in shape), "Expected all elements to be \"Expression\""
        return ir.NumericalType(data_type=data_type, shape=shape)

    def visitDtype(self, ctx: FhYParser.DtypeContext) -> Any:
        return ir.DataType(ir.PrimitiveDataType(ctx.IDENTIFIER().getText()))

    def visitIndex_type(self, ctx: FhYParser.Index_typeContext) -> Any:
        low, high, stride = self.visitRange(ctx.range_())
        assert isinstance(low, ast.Expression), f"Expected \"Expression\", got {type(low)}"
        assert isinstance(high, ast.Expression), f"Expected \"Expression\", got {type(high)}"
        if stride is not None:
            assert isinstance(stride, ast.Expression), f"Expected \"Expression\", got {type(stride)}"
        return ir.IndexType(lower_bound=low, upper_bound=high, stride=stride)

    def visitRange(self, ctx: FhYParser.RangeContext) -> Any:
        low_ctx = ctx.expression(0)
        low = self.visitExpression(low_ctx)
        assert isinstance(low, ast.Expression), f"Expected \"Expression\", got {type(low)}"

        high_ctx = ctx.expression(1)
        high = self.visitExpression(high_ctx)
        assert isinstance(high, ast.Expression), f"Expected \"Expression\", got {type(high)}"

        stride = None
        if (stride_ctx := ctx.expression(2)) is not None:
            stride = self.visitExpression(stride_ctx)
            assert isinstance(stride, ast.Expression), f"Expected \"Expression\", got {type(stride)}"

        return low, high, stride


# class __ParseTreeConverter(FhYListener):
#     # TODO Jason: Add docstring
#     _builder: ASTBuilder
#     _ast: Optional[ast.ASTNode]

#     def __init__(self, log: Logger = _log) -> None:
#         super().__init__()
#         self._builder = ASTBuilder()
#         self._ast = None
#         self._log = log

#     def _set_current_frame_span_for_current_context(
#         self, ctx: ParserRuleContext
#     ) -> None:
#         span = _get_source_info(ctx)
#         self._builder.set_current_frame_span(span)

#     def enterModule(self, ctx: FhYParser.ModuleContext) -> None:
#         self._builder.open_context(Module)

#     def exitModule(self, ctx: FhYParser.ModuleContext) -> None:
#         self._set_current_frame_span_for_current_context(ctx)
#         module_builder_frame = self._builder.close_context()
#         if not issubclass(module_builder_frame.cls, Module):
#             raise Exception()
#             # raise ContextError.message("module", Module.keyname(), module_builder_frame)
#         self._ast = module_builder_frame.build()

#     def enterComponent(self, ctx: FhYParser.ComponentContext) -> None:
#         if ctx.function_declaration() is not None:
#             raise NotImplementedError("Function declarations are not yet supported.")
#         elif ctx.function_definition() is not None:
#             pass
#         else:
#             raise NotImplementedError()

#     def exitComponent(self, ctx: FhYParser.ComponentContext) -> None:
#         def is_any_component_parsed() -> bool:
#             return any(
#                 [
#                     ctx.function_declaration(),
#                     ctx.function_definition(),
#                 ]
#             )

#         if is_any_component_parsed():
#             self._set_current_frame_span_for_current_context(ctx)
#             component_builder_frame: ASTBuilderFrame = self._builder.close_context()
#             if not issubclass(component_builder_frame.cls, Component):
#                 raise Exception()
#                 # raise ContextError.message("component", Component.keyname(), component_builder_frame)

#             module_builder_frame: ASTBuilderFrame = self._builder.get_current_frame()
#             if not issubclass(module_builder_frame.cls, Module):
#                 raise Exception()
#                 # raise ContextError.message("component", Module.keyname(), module_builder_frame)

#             module_builder_frame.components.append(component_builder_frame.build())

#     def enterFunction_header(self, ctx: FhYParser.Function_headerContext):
#         function_keyword: str = ctx.FUNCTION_KEYWORD().getText()

#         if function_keyword == "proc":
#             self._builder.open_context(Procedure)
#         elif function_keyword == "op":
#             self._builder.open_context(Operation)
#         else:
#             raise NotImplementedError()

#         function_name: str = ctx.IDENTIFIER().getText()
#         self._builder.get_current_frame().name = ir.Identifier(function_name)

#     def enterFunction_arg(self, ctx: FhYParser.Function_argContext):
#         if (arg_name := ctx.IDENTIFIER()) is None:
#             raise SyntaxError()

#         self._builder.open_context(Argument)
#         self._builder.get_current_frame().name = arg_name.getText()

#     def exitFunction_arg(self, ctx: FhYParser.Function_argContext):
#         self._set_current_frame_span_for_current_context(ctx)
#         argument_builder_frame: ASTBuilderFrame = self._builder.close_context()
#         if not issubclass(argument_builder_frame.cls, Argument):
#             raise Exception()
#             # raise ContextError.message("argument", Argument.keyname(), argument_builder_frame)

#         function_builder_frame: ASTBuilderFrame = self._builder.get_current_frame()
#         if not issubclass(function_builder_frame.cls, Function):
#             raise Exception()
#             # raise ContextError.message("argument", Function.keyname(), function_builder_frame)

#         function_builder_frame.args.append(argument_builder_frame.build())

#     # def enterFunction_body(self, ctx: FhYParser.Function_bodyContext):
#     #     self._log.debug("Enter")

#     # def exitFunction_body(self, ctx: FhYParser.Function_bodyContext):
#     #     self._log.debug("Exit")

#     def enterAtom(self, ctx: FhYParser.AtomContext):
#         self._log.debug("Enter")
#         if ctx.literal() is not None:
#             return
#         location: Span = _get_source_info(ctx)
#         self._builder.add_identifier(location, ctx.getText())

#     # STATEMENT CONTEXTS
#     def enterDeclaration_statement(self, ctx: FhYParser.Declaration_statementContext):
#         self._log.debug("Enter")
#         if (identifier := ctx.IDENTIFIER()) is None:
#             where = _get_source_info(ctx.parentCtx)
#             raise SyntaxError(
#                 f"No Identifier was provided for Declaration Statement: {where}"
#             )

#         name = identifier.getText()
#         location: Span = _get_source_info(ctx)
#         self._builder.open_declaration_statement(location, name)

#     def exitDeclaration_statement(self, ctx: FhYParser.Declaration_statementContext):
#         self._log.debug("Exit")
#         self._builder.close_declaration_statement()

#     def enterExpression_statement(self, ctx:FhYParser.Expression_statementContext):
#         self._log.debug("Enter")
#         location = _get_source_info(ctx)
#         name = ctx.IDENTIFIER()
#         if name is not None:
#             name = name.getText()
#         self._builder.open_expression_statement(location, name)

#     def exitExpression_statement(self, ctx:FhYParser.Expression_statementContext):
#         self._log.debug("Exit")
#         self._builder.close_expression_statement()

#     # def enterSelection_statement(self, ctx:FhYParser.Selection_statementContext):
#     #     # We have two Statement Children of a Selection Statement, which
#     #     # may or may not contain children themselves.
#     #     self._builder.open_branch_statement()

#     # def exitSelection_statement(self, ctx:FhYParser.Selection_statementContext):
#     #     self._builder.close_branch_statement()

#     def enterIteration_statement(self, ctx: FhYParser.Iteration_statementContext):
#         self._log.debug("Enter")
#         location: Span = _get_source_info(ctx)
#         self._builder.open_iteration_statement(location)

#     def exitIteration_statement(self, ctx: FhYParser.Iteration_statementContext):
#         self._log.debug("Exit")
#         self._builder.close_iteration_statement()

#     def enterReturn_statement(self, ctx: FhYParser.Return_statementContext):
#         self._log.debug("Enter")
#         location = _get_source_info(ctx)
#         self._builder.open_return_statement(location)

#     def exitReturn_statement(self, ctx: FhYParser.Return_statementContext):
#         self._log.debug("Exit")
#         self._builder.close_return_statement()

#     # TYPE CONTEXTS
#     def enterQualified_type(self, ctx: FhYParser.Qualified_typeContext):
#         self._builder.open_context(QualifiedType)

#         if (type_qualifier := ctx.IDENTIFIER()) is None:
#             raise SyntaxError()

#         self._builder.get_current_frame().type_qualifier = ir.TypeQualifier(
#             type_qualifier.getText()
#         )

#     def exitQualified_type(self, ctx: FhYParser.Qualified_typeContext):
#         self._set_current_frame_span_for_current_context(ctx)
#         self._builder.close_qualified_type_building()

#     # def enterType(self, ctx:FhYParser.TypeContext):
#     #     pass

#     # def exitType(self, ctx:FhYParser.TypeContext):
#     #     pass

#     # def enterTuple_type(self, ctx:FhYParser.Tuple_typeContext):
#     #     pass

#     # def exitTuple_type(self, ctx:FhYParser.Tuple_typeContext):
#     #     pass

#     def enterNumerical_type(self, ctx: FhYParser.Numerical_typeContext):
#         self._builder.open_context(ir.NumericalType)

#     # def exitNumerical_type(self, ctx:FhYParser.Numerical_typeContext):
#     #     ...

#     def enterDtype(self, ctx: FhYParser.DtypeContext):
#         self._log.debug("Enter")
#         numerical_type_name: str = ctx.IDENTIFIER().getText()
#         self._builder.add_dtype(numerical_type_name)

#     # def exitDtype(self, ctx:FhYParser.DtypeContext):
#     #     pass

#     # def enterShape(self, ctx: FhYParser.ShapeContext):
#     #     self._log.debug("Enter")
#     #     self._builder.open_shape()

#     # def exitShape(self, ctx: FhYParser.ShapeContext):
#     #     self._log.debug("Exit")
#     #     self._builder.close_shape()

#     def enterIndex_type(self, ctx: FhYParser.Index_typeContext):
#         self._builder.open_context(ir.IndexType)

#     # def exitIndex_type(self, ctx: FhYParser.Index_typeContext):
#     #     self._log.debug("Exit")
#     #     self._builder.close_index_type()

#     def exitType(self, ctx: FhYParser.TypeContext):
#         self._builder.close_type_building()

#     # EXPRESSION CONTEXTS
#     def enterExpression(self, ctx: FhYParser.ExpressionContext):
#         self._log.debug("Enter")
#         location = _get_source_info(ctx)
#         if ctx.primary_expression() is not None:
#             ...

#         elif ctx.nested_expression is not None:
#             # We pass since a nested expression contains a child expression.
#             # which will return back here anyways and be solved then.
#             ...

#         elif (unary := ctx.unary_expression) is not None:
#             self._builder.add_unary_expression(location, unary.text)

#         elif ctx.multiplicative_expression is not None:
#             op = ctx.DIVISION() or ctx.MULTIPLICATION()
#             self._builder.add_binary_expression(location, op.getText())

#         elif ctx.additive_expression is not None:
#             op = ctx.ADDITION() or ctx.SUBTRACTION()
#             self._builder.add_binary_expression(location, op.getText())

#         elif ctx.shift_expression is not None:
#             op = ctx.LEFT_SHIFT() or ctx.RIGHT_SHIFT()
#             self._builder.add_binary_expression(location, op.getText())

#         elif ctx.relational_expression is not None:
#             op = (
#                 ctx.LESS_THAN()
#                 or ctx.LESS_THAN_OR_EQUAL()
#                 or ctx.GREATER_THAN()
#                 or ctx.GREATER_THAN_OR_EQUAL()
#             )
#             self._builder.add_binary_expression(location, op.getText())

#         elif ctx.equality_expression is not None:
#             op = ctx.EQUAL_TO() or ctx.NOT_EQUAL_TO()
#             self._builder.add_binary_expression(location, op.getText())

#         elif ctx.and_expression is not None:
#             self._builder.add_binary_expression(location, ctx.AND().getText())

#         elif ctx.exclusive_or_expression is not None:
#             self._builder.add_binary_expression(location, ctx.EXCLUSIVE_OR().getText())

#         elif ctx.or_expression is not None:
#             self._builder.add_binary_expression(location, ctx.OR().getText())

#         elif ctx.logical_and_expression is not None:
#             self._builder.add_binary_expression(location, ctx.LOGICAL_AND().getText())

#         elif ctx.logical_or_expression is not None:
#             self._builder.add_binary_expression(location, ctx.LOGICAL_OR().getText())

#         elif ctx.ternary_expression is not None:
#             assert ctx.QUESTION_MARK().getText() == "?"
#             self._builder.open_ternary_expression(location)

#         else:
#             raise NotImplementedError("Unknown Expression Not Implemented")

#     def enterExpression_list(self, ctx:FhYParser.Expression_listContext):
#         self._log.debug("Enter")
#         self._builder.open_expression_list()

#     def exitExpression_list(self, ctx:FhYParser.Expression_listContext):
#         self._log.debug("Exit")
#         self._builder.close_expression_list()

#     def exitExpression(self, ctx: FhYParser.ExpressionContext):
#         self._log.debug("Exit")
#         if ctx.primary_expression() is not None:
#             ...

#         elif ctx.nested_expression is not None:
#             # We pass since a nested expression contains a child expression.
#             # which will return back here anyways and be solved then.
#             ...

#         elif ctx.unary_expression is not None:
#             self._builder.close_unary_expression()

#         elif any(
#             i is not None
#             for i in (
#                 ctx.multiplicative_expression,
#                 ctx.additive_expression,
#                 ctx.shift_expression,
#                 ctx.relational_expression,
#                 ctx.equality_expression,
#                 ctx.and_expression,
#                 ctx.or_expression,
#                 ctx.logical_and_expression,
#                 ctx.logical_or_expression,
#             )
#         ):
#             self._builder.close_binary_expression()

#         elif ctx.ternary_expression is not None:
#             self._builder.close_ternary_expression()

#         else:
#             raise NotImplementedError("Unknown Expression Not Implemented")

#     def enterPrimary_expression(self, ctx: FhYParser.Primary_expressionContext):

#         location = _get_source_info(ctx)
#         if ctx.tuple_access_expression is not None:
#             self._log.debug("Enter Tuple Access Expression")

#         elif ctx.function_expression is not None:
#             self._log.debug("Enter Function Expression")

#         elif ctx.tensor_access_expression is not None:
#             self._log.debug("Enter Tensor Access Expression")
#             self._builder.open_tensor_access_expression(location)

#         elif ctx.atom() is not None:
#             self._log.debug("Enter Atom Expression")
#             ...

#         else:
#             raise NotImplementedError("Unknown Primary Expression")

#     def exitPrimary_expression(self, ctx:FhYParser.Primary_expressionContext):
#         self._log.debug("Exit")

#     def exitStatement(self, ctx: FhYParser.StatementContext):
#         self._log.debug("Exit")
#         self._builder.close_statement()

#     # LITERALS
#     def enterLiteral(self, ctx: FhYParser.LiteralContext):
#         self._log.debug("Enter")
#         location = _get_source_info(ctx)
#         # TODO: Capturing Floats is Not Working Here...
#         if (floats := ctx.float_literal()) is not None:
#             value = float(floats.getText())

#         elif (integer := ctx.INT_LITERAL()) is not None:
#             value = int(integer.getText())

#         # TODO: Handle Complex Values
#         else:
#             raise NotImplementedError("Unknown Type Literal")
#         self._builder.add_literal(location, value)

#     @property
#     def ast(self) -> Optional[ASTNode]:
#         return self._ast


def from_parse_tree(parse_tree: FhYParser.ModuleContext) -> ast.ASTNode:
    # TODO Jason: Add docstring
    converter = ParseTreeConverter()
    # walker = ParseTreeWalker()
    # walker.walk(converter, parse_tree)
    _ast = converter.visitModule(parse_tree)
    # assert len(converter._builder._frame_stack) == 0, "Incomplete AST Build."
    if _ast is None:
        # TODO Jason: Implement a better error for this ast conversion failure
        raise Exception()

    return _ast
