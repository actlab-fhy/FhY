"""Tools to Construct an AST from a FhY Concrete Syntax Tree using Visitors.

Classes:
    ParseTreeConverter: Handles the actual construction of the AST from CST

Functions:
    from_parse_tree: Primary entry point to build an AST from a CST

"""

from collections import ChainMap
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from antlr4 import ParserRuleContext  # type: ignore[import-untyped]

from fhy import ir
from fhy.lang import ast
from fhy.lang.parser import FhYParser, FhYVisitor  # type: ignore[import-untyped]
from fhy.utils.alias import Expressions

from ...span import Span


def _get_source_info(ctx: ParserRuleContext, parent: bool = False) -> Span:
    """Retrieves line and column information from a context"""
    start = ctx.start
    stop = ctx.stop

    if all((start, stop)):
        return Span(start.line, stop.line, start.column, stop.column)

    elif not parent and (parent_ctx := getattr(ctx, "parentCtx", None)) is not None:
        return _get_source_info(parent_ctx, True)

    return Span(0, 0, 0, 0)


def _initialize_builtin_identifiers() -> Dict[str, ir.Identifier]:
    return {
        "sum": ir.Identifier("sum"),
    }


class ParseTreeConverter(FhYVisitor):
    """Constructs an AST representation from a FhY Concrete Syntax Tree Node Visitor.

    Notes:
        This class uses a visitor pattern to collect relevant information in the
        construction of ASTNode(s), by visiting relevant children in the concrete syntax
        tree. During Construction, we use a basic chainmap to primitively control basic
        scoping contexts. In particular, the scope is used to determine whether a
        variable has been previously declared before assigning an Identifier. Otherwise,
        a variable would would be assigned multiple IDs every time we encounter it
        (independent of scope).

    """

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

    # =====================
    # MODULE VISITORS
    # =====================
    def visitModule(self, ctx: FhYParser.ModuleContext) -> ast.Module:
        span = _get_source_info(ctx)

        self._open_scope()
        components: List[ast.Component] = []
        for component_ctx in ctx.component():
            component = self.visitComponent(component_ctx)
            components.append(component)
        self._close_scope()

        return ast.Module(components=components, span=span)

    # =====================
    # IMPORT VISITORS
    # =====================
    def visitImport_component(
        self, ctx: FhYParser.Import_componentContext
    ) -> ast.Import:
        identifier_expression_ctx: FhYParser.Identifier_expressionContext = (
            ctx.identifier_expression()
        )
        name_hint_components: list[str] = []
        for module_name in identifier_expression_ctx.IDENTIFIER():
            name_hint_components.append(module_name.getText())
        name_hint = ".".join(name_hint_components)
        span = _get_source_info(ctx)
        return ast.Import(name=self._get_identifier(name_hint), span=span)

    # =====================
    # FUNCTION VISITORS
    # =====================
    def visitFunction_declaration(
        self, ctx: FhYParser.Function_declarationContext
    ) -> Any:
        # TODO: Implement
        span = _get_source_info(ctx)
        line, col = span.line, span.column
        text = f"Lines {line.start}:{col.start} - {line.stop}:{col.stop}"

        raise NotImplementedError(f"Function Declarations are not Supported. {text}")

    def visitFunction_definition(
        self, ctx: FhYParser.Function_definitionContext
    ) -> Union[ast.Operation, ast.Procedure]:
        # TODO: add template types and indices (3rd and 4th returned values here)
        # TODO: consider getting function name here as the open scope needed to be moved
        #       to function header so the function name is still in the parent scope
        keyword, name, template, indices, args, return_type = self.visitFunction_header(
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
            if return_type is None:
                line, col = span.line, span.column
                text = f"Lines {line.start}:{col.start} - {line.stop}:{col.stop}"
                raise SyntaxError(f"No Operation Return Type Provided. {text}")

            return ast.Operation(
                name=name,
                args=args,
                return_type=return_type,
                body=body,
                span=span,
            )

        else:
            line, col = span.line, span.column
            text = f"Lines {line.start}:{col.start} - {line.stop}:{col.stop}"
            raise SyntaxError(f"Invalid Function Keyword Provided. {text}: {keyword}")

    def visitFunction_header(
        self, ctx: FhYParser.Function_headerContext
    ) -> Tuple[
        str,
        ir.Identifier,
        None,
        None,
        List[ast.Argument],
        Optional[ast.QualifiedType],
    ]:
        # TODO: If a Function Keyword is not Found, it is not parsed.
        #       Currently, We can Never get here
        if (kw_ctx := ctx.FUNCTION_KEYWORD()) is None:
            location = _get_source_info(ctx)
            line, col = location.line, location.column
            text = f"Lines {line.start}:{col.start} - {line.stop}:{col.stop}"
            raise SyntaxError(f"No Function Keyword Provided. {text}")
        keyword: str = kw_ctx.getText()

        # NOTE: This error is raised by Antlr before we get here
        if (name_ctx := ctx.IDENTIFIER()) is None:
            location = _get_source_info(ctx)
            line, col = location.line, location.column
            text = f"Lines {line.start}:{col.start} - {line.stop}:{col.stop}"
            raise SyntaxError(f"No Function Name Provided. {text}")

        name_hint: str = name_ctx.getText()
        name: ir.Identifier = self._get_identifier(name_hint)

        self._open_scope()

        args_ctx: FhYParser.Function_argsContext = ctx.function_args(0)
        args: List[ast.Argument] = self.visitFunction_args(args_ctx)

        # TODO: Implement Support for Function template and indices
        # template: List[ir.Identifier] = []
        # if (template_ctx := ctx.function_template_types) is not None:
        #     template.extend(self.visitIdentifier_list(template_ctx))

        # indices = List[ast.Argument] = []
        # if (index_ctx := ctx.function_indices) is not None:
        #     indices.extend(self.visitFunction_args(index_ctx))

        return_type: Optional[ast.QualifiedType] = None
        if (return_type_ctx := ctx.qualified_type()) is not None:
            return_type = self.visitQualified_type(return_type_ctx)

        # return keyword, name, template, indices, args, return_type
        return keyword, name, None, None, args, return_type

    def visitFunction_args(
        self, ctx: FhYParser.Function_argsContext
    ) -> List[ast.Argument]:
        args: List[ast.Argument] = []
        if ctx.function_arg() is not None:
            for arg_ctx in ctx.function_arg():
                arg: ast.Argument = self.visitFunction_arg(arg_ctx)
                args.append(arg)
        return args

    def visitFunction_arg(self, ctx: FhYParser.Function_argContext) -> ast.Argument:
        qualified_type = self.visitQualified_type(ctx.qualified_type())

        if (_id := ctx.IDENTIFIER()) is None:
            location = _get_source_info(ctx)
            line, col = location.line, location.column
            text = f"Lines {line.start}:{col.start} - {line.stop}:{col.stop}"
            raise SyntaxError(f"Function Argument Name not Provided. {text}")

        name_hint: str = _id.getText()
        name = self._get_identifier(name_hint)
        span = _get_source_info(ctx)

        return ast.Argument(qualified_type=qualified_type, name=name, span=span)

    def visitFunction_body(
        self, ctx: FhYParser.Function_bodyContext
    ) -> List[ast.Statement]:
        return self.visitStatement_series(ctx.statement_series())

    # =====================
    # STATEMENT VISITORS
    # =====================
    def visitStatement_series(
        self, ctx: FhYParser.Statement_seriesContext
    ) -> List[ast.Statement]:
        statements: List[ast.Statement] = []
        if ctx.statement() is not None:
            for statement_ctx in ctx.statement():
                statement = self.visitStatement(statement_ctx)
                statements.append(statement)

        return statements

    def visitDeclaration_statement(
        self, ctx: FhYParser.Declaration_statementContext
    ) -> ast.DeclarationStatement:
        qualified_type = self.visitQualified_type(ctx.qualified_type())

        # NOTE: This validation step is performed for type safety, but we will never
        #       actually get here, because this is a valid Expression Statement.
        if (_id := ctx.IDENTIFIER()) is None:
            location = _get_source_info(ctx)
            line, col = location.line, location.column
            text = f"Lines {line.start}:{col.start} - {line.stop}:{col.stop}"
            raise SyntaxError(f"Variable Name not Declared. {text}")

        name_hint: str = _id.getText()
        name = self._get_identifier(name_hint)
        expression = None
        if (expression_ctx := ctx.expression()) is not None:
            expression = self.visitExpression(expression_ctx)

        span: Span = _get_source_info(ctx)
        return ast.DeclarationStatement(
            variable_type=qualified_type,
            variable_name=name,
            expression=expression,
            span=span,
        )

    def visitExpression_statement(
        self, ctx: FhYParser.Expression_statementContext
    ) -> ast.ExpressionStatement:
        left_expression = None
        if (primitive_expression_ctx := ctx.primitive_expression()) is not None:
            left_expression = self.visitPrimitive_expression(primitive_expression_ctx)

        right_expression_ctx = ctx.expression()
        right_expression = self.visitExpression(right_expression_ctx)
        span = _get_source_info(ctx)

        return ast.ExpressionStatement(
            left=left_expression, right=right_expression, span=span
        )

    def visitSelection_statement(
        self, ctx: FhYParser.Selection_statementContext
    ) -> ast.SelectionStatement:
        condition_ctx = ctx.expression()
        condition = self.visitExpression(condition_ctx)

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

    def visitIteration_statement(
        self, ctx: FhYParser.Iteration_statementContext
    ) -> ast.ForAllStatement:
        index_ctx = ctx.expression()
        index = self.visitExpression(index_ctx)

        self._open_scope()
        body_ctx = ctx.statement_series()
        body = self.visitStatement_series(body_ctx)

        self._close_scope()
        span = _get_source_info(ctx)
        return ast.ForAllStatement(index=index, body=body, span=span)

    def visitReturn_statement(
        self, ctx: FhYParser.Return_statementContext
    ) -> ast.ReturnStatement:
        expression_ctx = ctx.expression()
        expression = self.visitExpression(expression_ctx)
        span = _get_source_info(ctx)

        return ast.ReturnStatement(expression=expression, span=span)

    # =====================
    # EXPRESSION VISITORS
    # =====================
    def visitExpression_list(
        self, ctx: FhYParser.Expression_listContext
    ) -> Sequence[Expressions]:
        expressions: List[Expressions] = []
        if ctx.expression() is not None:
            for expression_ctx in ctx.expression():
                expressions.append(self.visitExpression(expression_ctx))
        return expressions

    def visitExpression(self, ctx: FhYParser.ExpressionContext) -> ast.Expression:
        span = _get_source_info(ctx)
        if ctx.nested_expression is not None:
            return self.visitExpression(ctx.expression(0))

        elif ctx.unary_expression is not None:
            operand = self.visitExpression(ctx.expression(0))
            operator_ctx = ctx.SUBTRACTION() or ctx.BITWISE_NOT() or ctx.LOGICAL_NOT()

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
            right = self.visitExpression(ctx.expression(1))

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

            return ast.BinaryExpression(
                span=span,
                operation=ast.BinaryOperation(operator.getText()),
                left=left,
                right=right,
            )

        elif ctx.ternary_expression is not None:
            condition = self.visitExpression(ctx.expression(0))
            true_expression = self.visitExpression(ctx.expression(1))
            false_expression = self.visitExpression(ctx.expression(2))

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

            return primitive_expression

        else:
            line, col = span.line, span.column
            text = f"Lines {line.start}:{col.start} - {line.stop}:{col.stop}"
            raise NotImplementedError(f"Invalid Primitive Expression. {text}")

    def visitPrimitive_expression(
        self, ctx: FhYParser.Primitive_expressionContext
    ) -> ast.Expression:
        span = _get_source_info(ctx)
        # TODO: Add support of Tuple Access Expressions
        # if ctx.tuple_access_expression is not None:
        #     primitive_expression_ctx = ctx.primitive_expression()
        #     primitive_expression = self.visitPrimitive_expression(
        #         primitive_expression_ctx
        #     )

        #     return ast.TupleAccessExpression(
        #         span=span,
        #         tuple_expression=primitive_expression,
        #         element_index=int(ctx.INT_LITERAL().getText()),
        #     )

        if ctx.function_expression is not None:
            function_expression_ctx = ctx.primitive_expression()
            function_expression = self.visitPrimitive_expression(
                function_expression_ctx
            )

            expression_list_counter: int = 0

            template_types: List[ast.Expression] = []
            if ctx.LESS_THAN() is not None and ctx.GREATER_THAN() is not None:
                template_types = self.visitExpression_list(
                    ctx.expression_list(expression_list_counter)
                )
                expression_list_counter += 1

            indices: List[ast.Expression] = []
            if ctx.OPEN_BRACKET() is not None and ctx.CLOSE_BRACKET() is not None:
                indices = self.visitExpression_list(
                    ctx.expression_list(expression_list_counter)
                )
                expression_list_counter += 1

            args = self.visitExpression_list(
                ctx.expression_list(expression_list_counter)
            )

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
            indices_ctx = ctx.expression_list(0)
            indices = self.visitExpression_list(indices_ctx)

            return ast.ArrayAccessExpression(
                array_expression=array_expression, indices=indices, span=span
            )

        elif (atom_ctx := ctx.atom()) is not None:
            atom_expression = self.visitAtom(atom_ctx)

            return atom_expression

        else:
            # TODO: Replace with better Exception / reporting
            raise Exception("Invalid Primitive Expression")

    def visitIdentifier_expression(
        self, ctx: FhYParser.Identifier_expressionContext
    ) -> ast.IdentifierExpression:
        return ast.IdentifierExpression(
            identifier=self._get_identifier(ctx.getText()), span=_get_source_info(ctx)
        )

    def visitLiteral(self, ctx: FhYParser.LiteralContext) -> ast.Literal:
        span = _get_source_info(ctx)
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

            return ast.IntLiteral(span=span, value=int(int_literal_str, base=base))

        elif (float_literal_ctx := ctx.FLOAT_LITERAL()) is not None:
            float_literal = ast.FloatLiteral(
                span=span, value=float(float_literal_ctx.getText())
            )
            return float_literal

        else:
            line, col = span.line, span.column
            text = f"Lines {line.start}:{col.start} - {line.stop}:{col.stop}"
            raise NotImplementedError(f"Unsupported Type Literal. {text}")

    # =====================
    # TYPE VISITORS
    # =====================
    def visitQualified_type(
        self, ctx: FhYParser.Qualified_typeContext
    ) -> ast.QualifiedType:
        type_qualifier: Optional[ir.TypeQualifier] = None
        if (type_qualifier_ctx := ctx.IDENTIFIER()) is not None:
            type_qualifier = ir.TypeQualifier(type_qualifier_ctx.getText())

        else:
            # TODO: Do we expect to support unqualified types?
            #       Do we need a placeholder base qual to be replaced by analysis pass?
            # TODO: Replace this exception with better error handling.
            raise Exception("Invalid Type Qualifier")

        base_type = self.visitType(ctx.type_())
        span = _get_source_info(ctx)

        return ast.QualifiedType(
            base_type=base_type,
            type_qualifier=type_qualifier,
            span=span,
        )

    def visitNumerical_type(
        self, ctx: FhYParser.Numerical_typeContext
    ) -> ir.NumericalType:
        data_type = self.visitDtype(ctx.dtype())
        shape: Sequence[Expressions] = []
        if (shape_ctx := ctx.expression_list()) is not None:
            shape = self.visitExpression_list(shape_ctx)

        return ir.NumericalType(
            data_type=data_type,
            shape=list(shape),
        )

    def visitDtype(self, ctx: FhYParser.DtypeContext) -> ir.DataType:
        return ir.DataType(ir.PrimitiveDataType(ctx.IDENTIFIER().getText()))

    def visitIndex_type(self, ctx: FhYParser.Index_typeContext) -> ir.IndexType:
        low, high, stride = self.visitRange(ctx.range_())

        return ir.IndexType(lower_bound=low, upper_bound=high, stride=stride)

    def visitRange(
        self, ctx: FhYParser.RangeContext
    ) -> Tuple[ast.Expression, ast.Expression, Optional[ast.Expression]]:
        low_ctx = ctx.expression(0)
        low = self.visitExpression(low_ctx)
        high_ctx = ctx.expression(1)
        high = self.visitExpression(high_ctx)

        stride: Optional[ast.Expression] = None
        if (stride_ctx := ctx.expression(2)) is not None:
            stride = self.visitExpression(stride_ctx)

        return low, high, stride

    def visitTuple_type(self, ctx: FhYParser.Tuple_typeContext) -> ir.type.TupleType:
        types: List[ir.Type] = []
        if (context := ctx.type_()) is not None:
            for t in context:
                types.append(self.visitType(t))

        return ir.type.TupleType(types=types)


def from_parse_tree(parse_tree: FhYParser.ModuleContext) -> ast.Module:
    """Constructs an AST from a Concrete Syntax Tree."""
    converter = ParseTreeConverter()
    _ast: ast.Module = converter.visitModule(parse_tree)
    if _ast is None:
        # TODO: Raise Better, Custom Exception (e.g. FhYASTBuildError)
        raise Exception("AST Not Built")

    return _ast
