from typing import List, Optional

from antlr4 import ParseTreeWalker

from fhy.lang.ast import ASTNode
from fhy.lang.parser import FhYListener, FhYParser

from ..builder import ASTBuilder


class ParseTreeConverter(FhYListener):
    # TODO Jason: Add docstring
    _builder: ASTBuilder
    _ast: Optional[ASTNode]

    def __init__(self) -> None:
        super().__init__()
        self._builder = ASTBuilder()
        self._ast = None

    def enterModule(self, ctx: FhYParser.ModuleContext) -> None:
        self._builder.add_module()

    def exitModule(self, ctx: FhYParser.ModuleContext) -> None:
        self._builder.close_module_building()
        self._ast = self._builder.ast

    def enterComponent(self, ctx: FhYParser.ComponentContext) -> None:
        if ctx.function_declaration() is not None:
            raise NotImplementedError("Function Declarations are not yet Supported.")
        elif ctx.function_definition() is not None:
            pass
        else:
            raise NotImplementedError()

    def exitComponent(self, ctx: FhYParser.ComponentContext) -> None:
        if any([ctx.function_declaration(), ctx.function_definition()]):
            self._builder.close_component_building()

    def enterFunction_header(self, ctx: FhYParser.Function_headerContext):
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
        argument_name: str = ctx.IDENTIFIER().getText()
        self._builder.add_argument(argument_name)

    def exitFunction_arg(self, ctx: FhYParser.Function_argContext):
        self._builder.close_argument_building()

    # def enterFunction_body(self, ctx: FhYParser.Function_bodyContext):
    #     ...

    # def exitFunction_body(self, ctx: FhYParser.Function_bodyContext):
    #     ...

    # def enterAtom(self, ctx: FhYParser.AtomContext):
    #     ...

    # STATEMENT CONTEXTS
    def enterDeclaration_statement(self, ctx: FhYParser.Declaration_statementContext):
        name = ctx.IDENTIFIER().getText()
        self._builder.open_declaration_statement(name)

    def exitDeclaration_statement(self, ctx:FhYParser.Declaration_statementContext):
        self._builder.close_declaration_statement()

    # def enterExpression_statement(self, ctx:FhYParser.Expression_statementContext):
    #     pass

    # def exitExpression_statement(self, ctx:FhYParser.Expression_statementContext):
    #     pass

    # def enterSelection_statement(self, ctx:FhYParser.Selection_statementContext):
    #     pass

    # def exitSelection_statement(self, ctx:FhYParser.Selection_statementContext):
    #     pass

    # def enterIteration_statement(self, ctx:FhYParser.Iteration_statementContext):
    #     pass

    # def exitIteration_statement(self, ctx:FhYParser.Iteration_statementContext):
    #     pass

    def enterReturn_statement(self, ctx: FhYParser.Return_statementContext):
        self._builder.open_return_statement()

    def exitReturn_statement(self, ctx: FhYParser.Return_statementContext):
        self._builder.close_return_statement()

    # TYPE CONTEXTS
    def enterQualified_type(self, ctx: FhYParser.Qualified_typeContext):
        type_qualifier_name: str = ctx.IDENTIFIER().getText()
        self._builder.add_qualified_type(type_qualifier_name)

    def exitQualified_type(self, ctx: FhYParser.Qualified_typeContext):
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
        self._builder.add_numerical_type()

    # def exitNumerical_type(self, ctx:FhYParser.Numerical_typeContext):
    #     pass

    def enterDtype(self, ctx: FhYParser.DtypeContext):
        numerical_type_name: str = ctx.IDENTIFIER().getText()
        self._builder.add_dtype(numerical_type_name)

    # def exitDtype(self, ctx:FhYParser.DtypeContext):
    #     pass

    def enterShape(self, ctx: FhYParser.ShapeContext):
        # TODO: This is a little Wonky, and Won't support Expressions.
        shapes: List[str] = ctx.getText().split(",")
        self._builder.add_shape(shapes)

    # def exitShape(self, ctx:FhYParser.ShapeContext):
    #     pass

    def enterIndex_type(self, ctx: FhYParser.Index_typeContext):
        self._builder.add_index_type()

    # def exitIndex_type(self, ctx:FhYParser.Index_typeContext):
    #     pass

    def exitType(self, ctx: FhYParser.TypeContext):
        self._builder.close_type_building()

    # EXPRESSION CONTEXTS
    def enterExpression(self, ctx: FhYParser.ExpressionContext):

        if (primary := ctx.primary_expression()) is not None:
            ...

        elif ctx.nested_expression is not None:
            # We pass since a nested expression contains a child expression.
            # which will return back here anyways and be solved then.
            ...

        elif (unary := ctx.unary_expression) is not None:
            self._builder.add_unary_expression(unary.text)

        elif (mult := ctx.multiplicative_expression) is not None:
            op = ctx.DIVISION() or ctx.MULTIPLICATION()
            self._builder.add_binary_expression(op.getText())

        elif (add := ctx.additive_expression) is not None:
            op = ctx.ADDITION() or ctx.SUBTRACTION()
            self._builder.add_binary_expression(op.getText())

        elif (shift := ctx.shift_expression) is not None:
            op = ctx.LEFT_SHIFT() or ctx.RIGHT_SHIFT()
            self._builder.add_binary_expression(op.getText())

        elif (relate := ctx.relational_expression) is not None:
            op = ctx.LESS_THAN() or ctx.LESS_THAN_OR_EQUAL() or ctx.GREATER_THAN() or ctx.GREATER_THAN_OR_EQUAL()
            self._builder.add_binary_expression(op.getText())

        elif (equal := ctx.equality_expression) is not None:
            op = ctx.EQUAL_TO() or ctx.NOT_EQUAL_TO()
            self._builder.add_binary_expression(op.getText())

        elif (ands := ctx.and_expression) is not None:
            self._builder.add_binary_expression(ctx.AND().getText())

        elif (excl_or := ctx.exclusive_or_expression) is not None:
            self._builder.add_binary_expression(ctx.EXCLUSIVE_OR().getText())

        elif (ors := ctx.or_expression) is not None:
            self._builder.add_binary_expression(ctx.OR().getText())
    
        elif (logic_and := ctx.logical_and_expression) is not None:
            self._builder.add_binary_expression(ctx.LOGICAL_AND().getText())

        elif (logic_or := ctx.logical_or_expression) is not None:
            self._builder.add_binary_expression(ctx.LOGICAL_OR().getText())

        elif (ternary := ctx.ternary_expression) is not None:
            # value = ctx.QUESTION_MARK().getText()
            raise NotImplementedError("Ternary Expressions Not Implemented Yet.")

        else:
            raise NotImplementedError("Unknown Expression Not Implemented")

    def exitExpression(self, ctx: FhYParser.ExpressionContext):
        if (primary := ctx.primary_expression()) is not None:
            ...

        elif ctx.nested_expression is not None:
            # We pass since a nested expression contains a child expression.
            # which will return back here anyways and be solved then.
            ...

        elif ctx.unary_expression is not None:
            self._builder.close_unary_expression()

        elif ctx.multiplicative_expression is not None:
            self._builder.close_binary_expression()

        elif (add := ctx.additive_expression) is not None:
            self._builder.close_binary_expression()

        elif (shift := ctx.shift_expression) is not None:
            self._builder.close_binary_expression()

        elif (relate := ctx.relational_expression) is not None:
            self._builder.close_binary_expression()

        elif (equal := ctx.equality_expression) is not None:
            self._builder.close_binary_expression()

        elif (ands := ctx.and_expression) is not None:
            self._builder.close_binary_expression()
    
        elif (ors := ctx.or_expression) is not None:
            self._builder.close_binary_expression()
    
        elif (logic_and := ctx.logical_and_expression) is not None:
            self._builder.close_binary_expression()

        elif (logic_or := ctx.logical_or_expression) is not None:
            self._builder.close_binary_expression()

        elif (ternary := ctx.ternary_expression) is not None:
            raise NotImplementedError("Ternary Expressions Not Implemented Yet.")

        else:
            raise NotImplementedError("Unknown Expression Not Implemented")

    def enterPrimary_expression(self, ctx: FhYParser.Primary_expressionContext):
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
        self._builder.close_statement()

    # LITERALS
    def enterLiteral(self, ctx:FhYParser.LiteralContext):
        # TODO: Capturing Floats is Not Working Here...
        print("Entering Literal:", ctx.getText())
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
