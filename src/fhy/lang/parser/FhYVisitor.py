# Generated from grammar/FhY.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .FhYParser import FhYParser
else:
    from FhYParser import FhYParser

# This class defines a complete generic visitor for a parse tree produced by FhYParser.

class FhYVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by FhYParser#module.
    def visitModule(self, ctx:FhYParser.ModuleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#statement.
    def visitStatement(self, ctx:FhYParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#scope.
    def visitScope(self, ctx:FhYParser.ScopeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#import_statement.
    def visitImport_statement(self, ctx:FhYParser.Import_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#function_declaration.
    def visitFunction_declaration(self, ctx:FhYParser.Function_declarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#function_definition.
    def visitFunction_definition(self, ctx:FhYParser.Function_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#function_header.
    def visitFunction_header(self, ctx:FhYParser.Function_headerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#identifier_list.
    def visitIdentifier_list(self, ctx:FhYParser.Identifier_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#function_args.
    def visitFunction_args(self, ctx:FhYParser.Function_argsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#function_arg.
    def visitFunction_arg(self, ctx:FhYParser.Function_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#function_body.
    def visitFunction_body(self, ctx:FhYParser.Function_bodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#declaration_statement.
    def visitDeclaration_statement(self, ctx:FhYParser.Declaration_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#expression_statement.
    def visitExpression_statement(self, ctx:FhYParser.Expression_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#selection_statement.
    def visitSelection_statement(self, ctx:FhYParser.Selection_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#iteration_statement.
    def visitIteration_statement(self, ctx:FhYParser.Iteration_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#return_statement.
    def visitReturn_statement(self, ctx:FhYParser.Return_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#qualified_type.
    def visitQualified_type(self, ctx:FhYParser.Qualified_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#type.
    def visitType(self, ctx:FhYParser.TypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#tuple_type.
    def visitTuple_type(self, ctx:FhYParser.Tuple_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#numerical_type.
    def visitNumerical_type(self, ctx:FhYParser.Numerical_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#dtype.
    def visitDtype(self, ctx:FhYParser.DtypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#index_type.
    def visitIndex_type(self, ctx:FhYParser.Index_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#range.
    def visitRange(self, ctx:FhYParser.RangeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#expression.
    def visitExpression(self, ctx:FhYParser.ExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#expression_list.
    def visitExpression_list(self, ctx:FhYParser.Expression_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#primitive_expression.
    def visitPrimitive_expression(self, ctx:FhYParser.Primitive_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#atom.
    def visitAtom(self, ctx:FhYParser.AtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#tuple.
    def visitTuple(self, ctx:FhYParser.TupleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#identifier_expression.
    def visitIdentifier_expression(self, ctx:FhYParser.Identifier_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FhYParser#literal.
    def visitLiteral(self, ctx:FhYParser.LiteralContext):
        return self.visitChildren(ctx)



del FhYParser