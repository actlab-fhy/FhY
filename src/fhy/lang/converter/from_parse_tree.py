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

"""Tools to construct an AST from a FhY concrete syntax tree using visitors.

Classes:
    ParseTreeConverter: Handles the actual construction of the AST from CST

Functions:
    from_parse_tree: Primary entry point to build an AST from a CST

"""

import re
from collections import ChainMap
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from antlr4 import ParserRuleContext  # type: ignore[import-untyped]

from fhy import ir
from fhy.error import FhYASTBuildError, FhYSyntaxError
from fhy.lang import ast
from fhy.lang.ast import Span
from fhy.lang.ast.alias import Expressions
from fhy.lang.parser import FhYParser, FhYVisitor  # type: ignore[import-untyped]


def _get_source_info(ctx: ParserRuleContext, parent: bool = False) -> Span:
    """Retrieves line and column information from a context."""
    start = ctx.start
    stop = ctx.stop

    if all((start, stop)):
        return Span(start.line, stop.line, start.column, stop.column)

    elif not parent and (parent_ctx := getattr(ctx, "parentCtx", None)) is not None:
        return _get_source_info(parent_ctx, True)

    return Span(0, 0, 0, 0)


def _source_position(span: Span) -> str:
    line, col = span.line, span.column
    text = f"Lines {line.start}:{col.start} - {line.stop}:{col.stop}"

    return text


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

        statements: List[ast.Statement] = self.visitScope(ctx.scope())

        return ast.Module(statements=statements, span=span)

    # =====================
    # STATEMENT VISITORS
    # =====================
    def visitImport_statement(
        self, ctx: FhYParser.Import_statementContext
    ) -> ast.Import:
        identifier_expression_ctx: FhYParser.Identifier_expressionContext = (
            ctx.identifier_expression()
        )
        name_hint_components: list[str] = []
        for module_name in identifier_expression_ctx.IDENTIFIER():
            name_hint_components.append(module_name.getText())
        name_hint: str = ".".join(name_hint_components)
        span: Span = _get_source_info(ctx)

        return ast.Import(name=self._get_identifier(name_hint), span=span)

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

        body_ctx: FhYParser.Function_bodyContext = ctx.function_body()
        body: List[ast.Statement] = self.visitFunction_body(body_ctx)
        span: Span = _get_source_info(ctx)

        self._close_scope()

        if keyword == "proc":
            if return_type is not None:
                pos: Span = _get_source_info(ctx.function_header().qualified_type())
                text: str = _source_position(pos)
                raise FhYSyntaxError(f"Procedures do not have return types. {text}")

            return ast.Procedure(
                name=name, templates=template, args=args, body=body, span=span
            )

        elif keyword == "op":
            if return_type is None:
                pos = _get_source_info(ctx.function_header())
                text = _source_position(pos)
                raise FhYSyntaxError(
                    f"Operation Functions Require Return Types. {text}"
                )

            return ast.Operation(
                span=span,
                name=name,
                templates=template,
                args=args,
                body=body,
                return_type=return_type,
            )

        else:
            # NOTE: Defined Function Keywords are required by Antlr to parse the source
            #       code to meet the classification of "Function_definition". Meaning,
            #       we have no way to reach this code. Out of an abundance of caution:
            text = _source_position(span)
            raise FhYSyntaxError(
                f"Invalid Function Keyword Provided. {text}: {keyword}"
            )

    def visitFunction_header(
        self, ctx: FhYParser.Function_headerContext
    ) -> Tuple[
        str,
        ir.Identifier,
        List[ir.Identifier],
        List[ast.Argument],
        List[ast.Argument],
        Optional[ast.QualifiedType],
    ]:
        span: Span = _get_source_info(ctx)

        # NOTE: Predefined Function Keywords required for parsing Function.
        if (kw_ctx := ctx.FUNCTION_KEYWORD()) is None:
            text: str = _source_position(span)
            raise FhYSyntaxError(f"Function Keyword Missing. {text}")
        keyword: str = kw_ctx.getText()

        # NOTE: This error is raised by Antlr during construction of CST.
        if (name_ctx := ctx.IDENTIFIER()) is None:
            text = _source_position(span)
            raise FhYSyntaxError(f"Function Name Missing. {text}")

        name_hint: str = name_ctx.getText()
        name: ir.Identifier = self._get_identifier(name_hint)

        self._open_scope()

        args_ctx: FhYParser.Function_argsContext = ctx.function_args(0)
        args: List[ast.Argument] = self.visitFunction_args(args_ctx)

        # TODO: Implement Support for Function template and indices
        template: List[ir.Identifier] = []
        if ctx.function_template_types is not None:
            template_ctx: FhYParser.Identifier_listContext = ctx.identifier_list()
            template.extend(self.visitIdentifier_list(template_ctx))

        indices: List[ast.Argument] = []
        if (index_ctx := ctx.function_indices) is not None:
            indices.extend(self.visitFunction_args(index_ctx))

        return_type: Optional[ast.QualifiedType] = None
        if (return_type_ctx := ctx.qualified_type()) is not None:
            return_type = self.visitQualified_type(return_type_ctx)

        return keyword, name, template, indices, args, return_type

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
        span: Span = _get_source_info(ctx)

        qualified_type: ast.QualifiedType
        qualified_type = self.visitQualified_type(ctx.qualified_type())
        if (_id := ctx.IDENTIFIER()) is None:
            text = _source_position(span)
            raise FhYSyntaxError(f"Function Argument Name not Provided. {text}")

        name_hint: str = _id.getText()
        name = self._get_identifier(name_hint)

        return ast.Argument(qualified_type=qualified_type, name=name, span=span)

    def visitFunction_body(
        self, ctx: FhYParser.Function_bodyContext
    ) -> List[ast.Statement]:
        return self.visitScope(ctx.scope())

    def visitScope(self, ctx: FhYParser.ScopeContext) -> List[ast.Statement]:
        self._open_scope()
        statements: List[ast.Statement] = []
        if ctx.statement() is not None:
            for statement_ctx in ctx.statement():
                statement = self.visitStatement(statement_ctx)
                statements.append(statement)
        self._close_scope()

        return statements

    def visitDeclaration_statement(
        self, ctx: FhYParser.Declaration_statementContext
    ) -> ast.DeclarationStatement:
        span: Span = _get_source_info(ctx)
        qualified_type: ast.QualifiedType
        qualified_type = self.visitQualified_type(ctx.qualified_type())

        # NOTE: This validation step is performed for type safety. A statement without
        #       an Identifier would make a valid Expression Statement.
        if (_id := ctx.IDENTIFIER()) is None:
            text: str = _source_position(span)
            raise FhYSyntaxError(f"Variable Name not Declared. {text}")

        name_hint: str = _id.getText()
        name: ir.Identifier = self._get_identifier(name_hint)
        expression = None
        if (expression_ctx := ctx.expression()) is not None:
            expression = self.visitExpression(expression_ctx)

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

        right_expression_ctx: FhYParser.ExpressionContext = ctx.expression()
        right_expression: ast.Expression = self.visitExpression(right_expression_ctx)
        span: Span = _get_source_info(ctx)

        return ast.ExpressionStatement(
            left=left_expression, right=right_expression, span=span
        )

    def visitSelection_statement(
        self, ctx: FhYParser.Selection_statementContext
    ) -> ast.SelectionStatement:
        span: Span = _get_source_info(ctx)
        condition_ctx: FhYParser.ExpressionContext = ctx.expression()
        condition: ast.Expression = self.visitExpression(condition_ctx)

        true_body_ctx: FhYParser.ScopeContext = ctx.scope(0)
        true_body: List[ast.Statement] = self.visitScope(true_body_ctx)

        false_body: List[ast.Statement] = []
        if (false_body_ctx := ctx.scope(1)) is not None:
            false_body = self.visitScope(false_body_ctx)

        return ast.SelectionStatement(
            condition=condition, true_body=true_body, false_body=false_body, span=span
        )

    def visitIteration_statement(
        self, ctx: FhYParser.Iteration_statementContext
    ) -> ast.ForAllStatement:
        span: Span = _get_source_info(ctx)
        index_ctx: FhYParser.ExpressionContext = ctx.expression()
        index: ast.Expression = self.visitExpression(index_ctx)

        body_ctx: FhYParser.ScopeContext = ctx.scope()
        body: List[ast.Statement] = self.visitScope(body_ctx)

        return ast.ForAllStatement(index=index, body=body, span=span)

    def visitReturn_statement(
        self, ctx: FhYParser.Return_statementContext
    ) -> ast.ReturnStatement:
        span: Span = _get_source_info(ctx)
        expression_ctx: FhYParser.ExpressionContext = ctx.expression()
        expression: ast.Expression = self.visitExpression(expression_ctx)

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
        span: Span = _get_source_info(ctx)
        if ctx.nested_expression is not None:
            return self.visitExpression(ctx.expression(0))

        elif ctx.unary_expression is not None:
            operand: ast.Expression = self.visitExpression(ctx.expression(0))
            operator_ctx = ctx.SUBTRACTION() or ctx.BITWISE_NOT() or ctx.LOGICAL_NOT()

            return ast.UnaryExpression(
                span=span,
                operation=ast.UnaryOperation(operator_ctx.getText()),
                expression=operand,
            )

        elif any(
            [
                ctx.power_expression,
                ctx.multiplicative_expression,
                ctx.additive_expression,
                ctx.shift_expression,
                ctx.relational_expression,
                ctx.equality_expression,
                ctx.and_expression,
                ctx.or_expression,
                ctx.logical_and_expression,
                ctx.logical_or_expression,
                ctx.exclusive_or_expression,
            ]
        ):
            left: ast.Expression = self.visitExpression(ctx.expression(0))
            right: ast.Expression = self.visitExpression(ctx.expression(1))

            operator = (
                ctx.POWER()
                or ctx.MULTIPLICATION()
                or ctx.DIVISION()
                or ctx.FLOORDIV()
                or ctx.MODULO()
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
                or ctx.EXCLUSIVE_OR()
            )

            return ast.BinaryExpression(
                span=span,
                operation=ast.BinaryOperation(operator.getText()),
                left=left,
                right=right,
            )

        elif ctx.ternary_expression is not None:
            condition: ast.Expression = self.visitExpression(ctx.expression(0))
            true_expression: ast.Expression = self.visitExpression(ctx.expression(1))
            false_expression: ast.Expression = self.visitExpression(ctx.expression(2))

            return ast.TernaryExpression(
                span=span,
                condition=condition,
                true=true_expression,
                false=false_expression,
            )

        elif (primitive_expression_ctx := ctx.primitive_expression()) is not None:
            primitive_expression: ast.Expression = self.visitPrimitive_expression(
                primitive_expression_ctx
            )

            return primitive_expression

        else:
            text = _source_position(span)
            raise FhYSyntaxError(f"Invalid Primitive Expression. {text}")

    def visitPrimitive_expression(
        self, ctx: FhYParser.Primitive_expressionContext
    ) -> ast.Expression:
        span: Span = _get_source_info(ctx)
        if ctx.tuple_access_expression is not None:
            expression: ast.Expression = self.visitPrimitive_expression(
                ctx.primitive_expression()
            )

            # Grammar Hack to support Tuple Indexers require validation
            index_text: str = ctx.FLOAT_LITERAL().getText()
            if not index_text.startswith(".") or not re.search(
                r"^\.[0-9][0-9_]*$", index_text
            ):
                lines = _source_position(span)
                raise FhYSyntaxError(f'Invalid Tuple Accessor "{index_text}": {lines}')

            return ast.TupleAccessExpression(
                span=span,
                tuple_expression=expression,
                element_index=int(index_text[1:]),
            )

        elif ctx.function_expression is not None:
            function_expression_ctx: FhYParser.Primitive_expressionContext = (
                ctx.primitive_expression()
            )
            function_expression: ast.Expression = self.visitPrimitive_expression(
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

            args: List[ast.Expression] = self.visitExpression_list(
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
            array_expression_ctx: FhYParser.Primitive_expressionContext = (
                ctx.primitive_expression()
            )
            array_expression: ast.Expression = self.visitPrimitive_expression(
                array_expression_ctx
            )
            indices_ctx: FhYParser.Expression_listContext = ctx.expression_list(0)
            indices = self.visitExpression_list(indices_ctx)

            return ast.ArrayAccessExpression(
                array_expression=array_expression, indices=indices, span=span
            )

        elif (atom_ctx := ctx.atom()) is not None:
            atom_expression = self.visitAtom(atom_ctx)

            return atom_expression

        else:
            text: str = _source_position(span)
            raise FhYSyntaxError(f"Invalid Primitive Expression. {text}")

    def visitIdentifier_expression(
        self, ctx: FhYParser.Identifier_expressionContext
    ) -> ast.IdentifierExpression:
        return ast.IdentifierExpression(
            identifier=self._get_identifier(ctx.getText()), span=_get_source_info(ctx)
        )

    def visitIdentifier_list(
        self, ctx: FhYParser.Identifier_listContext
    ) -> List[ir.Identifier]:
        ids: List[ir.Identifier] = []
        for name in ctx.IDENTIFIER():
            ids.append(self._get_identifier(name.getText()))

        return ids

    def visitLiteral(self, ctx: FhYParser.LiteralContext) -> ast.Literal:
        span: Span = _get_source_info(ctx)
        if (int_literal_ctx := ctx.INT_LITERAL()) is not None:
            int_literal_str: str = int_literal_ctx.getText()

            if int_literal_str.startswith(("0x", "0X")):
                base: int = 16
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
            text = _source_position(span)
            raise NotImplementedError(f"Unsupported Type Literal. {text}")

    # =====================
    # TYPE VISITORS
    # =====================
    def visitQualified_type(
        self, ctx: FhYParser.Qualified_typeContext
    ) -> ast.QualifiedType:
        span: Span = _get_source_info(ctx)

        type_qualifier: Optional[ir.TypeQualifier] = None
        if (type_qualifier_ctx := ctx.IDENTIFIER()) is not None:
            type_qualifier = ir.TypeQualifier(type_qualifier_ctx.getText())

        else:
            text: str = _source_position(span)
            raise FhYSyntaxError(f"No Type Qualifier Provided. {text}")

        base_type = self.visitType(ctx.type_())

        return ast.QualifiedType(
            base_type=base_type,
            type_qualifier=type_qualifier,
            span=span,
        )

    def visitNumerical_type(
        self, ctx: FhYParser.Numerical_typeContext
    ) -> ir.NumericalType:
        data_type: ir.DataType = self.visitDtype(ctx.dtype())
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
        low_ctx: FhYParser.ExpressionContext = ctx.expression(0)
        low: ast.Expression = self.visitExpression(low_ctx)

        high_ctx: FhYParser.ExpressionContext = ctx.expression(1)
        high: ast.Expression = self.visitExpression(high_ctx)

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
    """Constructs an AST from a concrete syntax tree.

    Args:
        parse_tree (FhYParser.ModuleContext): FhY concrete syntax tree, module context.

    Raises:
        NotImplementedError: Attempted use of unsupported features of FhY language.
        FhYSyntaxError: Syntax error(s) found in FhY source code.
        FhYASTBuildError: AST failed to build from CST. Exact reason unknown.

    """
    converter = ParseTreeConverter()
    _ast: ast.Module = converter.visitModule(parse_tree)
    if _ast is None:
        raise FhYASTBuildError("Failed to Build AST from Concrete Syntax Tree.")

    return _ast
