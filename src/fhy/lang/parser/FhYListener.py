# Generated from grammar/FhY.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .FhYParser import FhYParser
else:
    from FhYParser import FhYParser

# This class defines a complete listener for a parse tree produced by FhYParser.
class FhYListener(ParseTreeListener):

    # Enter a parse tree produced by FhYParser#module.
    def enterModule(self, ctx:FhYParser.ModuleContext):
        pass

    # Exit a parse tree produced by FhYParser#module.
    def exitModule(self, ctx:FhYParser.ModuleContext):
        pass


    # Enter a parse tree produced by FhYParser#statement.
    def enterStatement(self, ctx:FhYParser.StatementContext):
        pass

    # Exit a parse tree produced by FhYParser#statement.
    def exitStatement(self, ctx:FhYParser.StatementContext):
        pass


    # Enter a parse tree produced by FhYParser#scope.
    def enterScope(self, ctx:FhYParser.ScopeContext):
        pass

    # Exit a parse tree produced by FhYParser#scope.
    def exitScope(self, ctx:FhYParser.ScopeContext):
        pass


    # Enter a parse tree produced by FhYParser#import_statement.
    def enterImport_statement(self, ctx:FhYParser.Import_statementContext):
        pass

    # Exit a parse tree produced by FhYParser#import_statement.
    def exitImport_statement(self, ctx:FhYParser.Import_statementContext):
        pass


    # Enter a parse tree produced by FhYParser#function_declaration.
    def enterFunction_declaration(self, ctx:FhYParser.Function_declarationContext):
        pass

    # Exit a parse tree produced by FhYParser#function_declaration.
    def exitFunction_declaration(self, ctx:FhYParser.Function_declarationContext):
        pass


    # Enter a parse tree produced by FhYParser#function_definition.
    def enterFunction_definition(self, ctx:FhYParser.Function_definitionContext):
        pass

    # Exit a parse tree produced by FhYParser#function_definition.
    def exitFunction_definition(self, ctx:FhYParser.Function_definitionContext):
        pass


    # Enter a parse tree produced by FhYParser#function_header.
    def enterFunction_header(self, ctx:FhYParser.Function_headerContext):
        pass

    # Exit a parse tree produced by FhYParser#function_header.
    def exitFunction_header(self, ctx:FhYParser.Function_headerContext):
        pass


    # Enter a parse tree produced by FhYParser#identifier_list.
    def enterIdentifier_list(self, ctx:FhYParser.Identifier_listContext):
        pass

    # Exit a parse tree produced by FhYParser#identifier_list.
    def exitIdentifier_list(self, ctx:FhYParser.Identifier_listContext):
        pass


    # Enter a parse tree produced by FhYParser#function_args.
    def enterFunction_args(self, ctx:FhYParser.Function_argsContext):
        pass

    # Exit a parse tree produced by FhYParser#function_args.
    def exitFunction_args(self, ctx:FhYParser.Function_argsContext):
        pass


    # Enter a parse tree produced by FhYParser#function_arg.
    def enterFunction_arg(self, ctx:FhYParser.Function_argContext):
        pass

    # Exit a parse tree produced by FhYParser#function_arg.
    def exitFunction_arg(self, ctx:FhYParser.Function_argContext):
        pass


    # Enter a parse tree produced by FhYParser#function_body.
    def enterFunction_body(self, ctx:FhYParser.Function_bodyContext):
        pass

    # Exit a parse tree produced by FhYParser#function_body.
    def exitFunction_body(self, ctx:FhYParser.Function_bodyContext):
        pass


    # Enter a parse tree produced by FhYParser#declaration_statement.
    def enterDeclaration_statement(self, ctx:FhYParser.Declaration_statementContext):
        pass

    # Exit a parse tree produced by FhYParser#declaration_statement.
    def exitDeclaration_statement(self, ctx:FhYParser.Declaration_statementContext):
        pass


    # Enter a parse tree produced by FhYParser#expression_statement.
    def enterExpression_statement(self, ctx:FhYParser.Expression_statementContext):
        pass

    # Exit a parse tree produced by FhYParser#expression_statement.
    def exitExpression_statement(self, ctx:FhYParser.Expression_statementContext):
        pass


    # Enter a parse tree produced by FhYParser#selection_statement.
    def enterSelection_statement(self, ctx:FhYParser.Selection_statementContext):
        pass

    # Exit a parse tree produced by FhYParser#selection_statement.
    def exitSelection_statement(self, ctx:FhYParser.Selection_statementContext):
        pass


    # Enter a parse tree produced by FhYParser#iteration_statement.
    def enterIteration_statement(self, ctx:FhYParser.Iteration_statementContext):
        pass

    # Exit a parse tree produced by FhYParser#iteration_statement.
    def exitIteration_statement(self, ctx:FhYParser.Iteration_statementContext):
        pass


    # Enter a parse tree produced by FhYParser#return_statement.
    def enterReturn_statement(self, ctx:FhYParser.Return_statementContext):
        pass

    # Exit a parse tree produced by FhYParser#return_statement.
    def exitReturn_statement(self, ctx:FhYParser.Return_statementContext):
        pass


    # Enter a parse tree produced by FhYParser#qualified_type.
    def enterQualified_type(self, ctx:FhYParser.Qualified_typeContext):
        pass

    # Exit a parse tree produced by FhYParser#qualified_type.
    def exitQualified_type(self, ctx:FhYParser.Qualified_typeContext):
        pass


    # Enter a parse tree produced by FhYParser#type.
    def enterType(self, ctx:FhYParser.TypeContext):
        pass

    # Exit a parse tree produced by FhYParser#type.
    def exitType(self, ctx:FhYParser.TypeContext):
        pass


    # Enter a parse tree produced by FhYParser#tuple_type.
    def enterTuple_type(self, ctx:FhYParser.Tuple_typeContext):
        pass

    # Exit a parse tree produced by FhYParser#tuple_type.
    def exitTuple_type(self, ctx:FhYParser.Tuple_typeContext):
        pass


    # Enter a parse tree produced by FhYParser#numerical_type.
    def enterNumerical_type(self, ctx:FhYParser.Numerical_typeContext):
        pass

    # Exit a parse tree produced by FhYParser#numerical_type.
    def exitNumerical_type(self, ctx:FhYParser.Numerical_typeContext):
        pass


    # Enter a parse tree produced by FhYParser#dtype.
    def enterDtype(self, ctx:FhYParser.DtypeContext):
        pass

    # Exit a parse tree produced by FhYParser#dtype.
    def exitDtype(self, ctx:FhYParser.DtypeContext):
        pass


    # Enter a parse tree produced by FhYParser#index_type.
    def enterIndex_type(self, ctx:FhYParser.Index_typeContext):
        pass

    # Exit a parse tree produced by FhYParser#index_type.
    def exitIndex_type(self, ctx:FhYParser.Index_typeContext):
        pass


    # Enter a parse tree produced by FhYParser#range.
    def enterRange(self, ctx:FhYParser.RangeContext):
        pass

    # Exit a parse tree produced by FhYParser#range.
    def exitRange(self, ctx:FhYParser.RangeContext):
        pass


    # Enter a parse tree produced by FhYParser#expression.
    def enterExpression(self, ctx:FhYParser.ExpressionContext):
        pass

    # Exit a parse tree produced by FhYParser#expression.
    def exitExpression(self, ctx:FhYParser.ExpressionContext):
        pass


    # Enter a parse tree produced by FhYParser#expression_list.
    def enterExpression_list(self, ctx:FhYParser.Expression_listContext):
        pass

    # Exit a parse tree produced by FhYParser#expression_list.
    def exitExpression_list(self, ctx:FhYParser.Expression_listContext):
        pass


    # Enter a parse tree produced by FhYParser#primitive_expression.
    def enterPrimitive_expression(self, ctx:FhYParser.Primitive_expressionContext):
        pass

    # Exit a parse tree produced by FhYParser#primitive_expression.
    def exitPrimitive_expression(self, ctx:FhYParser.Primitive_expressionContext):
        pass


    # Enter a parse tree produced by FhYParser#atom.
    def enterAtom(self, ctx:FhYParser.AtomContext):
        pass

    # Exit a parse tree produced by FhYParser#atom.
    def exitAtom(self, ctx:FhYParser.AtomContext):
        pass


    # Enter a parse tree produced by FhYParser#tuple.
    def enterTuple(self, ctx:FhYParser.TupleContext):
        pass

    # Exit a parse tree produced by FhYParser#tuple.
    def exitTuple(self, ctx:FhYParser.TupleContext):
        pass


    # Enter a parse tree produced by FhYParser#identifier_expression.
    def enterIdentifier_expression(self, ctx:FhYParser.Identifier_expressionContext):
        pass

    # Exit a parse tree produced by FhYParser#identifier_expression.
    def exitIdentifier_expression(self, ctx:FhYParser.Identifier_expressionContext):
        pass


    # Enter a parse tree produced by FhYParser#literal.
    def enterLiteral(self, ctx:FhYParser.LiteralContext):
        pass

    # Exit a parse tree produced by FhYParser#literal.
    def exitLiteral(self, ctx:FhYParser.LiteralContext):
        pass



del FhYParser