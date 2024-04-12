from typing import Optional
from antlr4 import ParseTreeWalker
from fhy.lang.parser import FhYListener, FhYParser
from fhy.lang.ast import ASTNode
from ..builder import ASTArgumentBuilder, ASTBuilder, ASTFunctionBuilder, ASTNumericalTypeBuilder, ASTQualifiedTypeBuilder


class ParseTreeConverter(FhYListener):
    # TODO Jason: Add docstring
    _builder: ASTBuilder
    _ast: Optional[ASTNode]

    def __init__(self) -> None:
        super().__init__()
        self._builder = ASTBuilder()
        self._ast = None

    def enterModule(self, ctx: FhYParser.ModuleContext) -> None:
        self._builder.open_module_context()

    def exitModule(self, ctx: FhYParser.ModuleContext) -> None:
        self._builder.close_module_context()
        self._ast = self._builder.ast

    def enterComponent(self, ctx: FhYParser.ComponentContext) -> None:
        if ctx.function_declaration() is not None:
            raise NotImplementedError()
        elif ctx.function_definition() is not None:
            pass
        else:
            raise NotImplementedError()

    def exitComponent(self, ctx: FhYParser.ComponentContext) -> None:
        if any([ctx.function_declaration(), ctx.function_definition()]):
            self._builder.close_component_context()

    def enterFunction_header(self, ctx: FhYParser.Function_headerContext):
        function_keyword: str = ctx.FUNCTION_KEYWORD().getText()
        if function_keyword == "proc":
            self._builder.open_procedure_context()
        elif function_keyword == "op":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        function_builder: ASTFunctionBuilder = self._builder.get_current_builder()
        function_name: str = ctx.IDENTIFIER().getText()
        function_builder.set_name_hint(function_name)

    def enterFunction_arg(self, ctx: FhYParser.Function_argContext):
        self._builder.open_argument_context()

        argument_builder: ASTArgumentBuilder = self._builder.get_current_builder()
        argument_name: str = ctx.IDENTIFIER().getText()
        argument_builder.set_name_hint(argument_name)

    def exitFunction_arg(self, ctx: FhYParser.Function_argContext):
        self._builder.close_argument_context()

    def enterQualified_type(self, ctx: FhYParser.Qualified_typeContext):
        self._builder.open_qualified_type_context()

        qualified_type_builder: ASTQualifiedTypeBuilder = self._builder.get_current_builder()
        type_qualifier_name: str = ctx.IDENTIFIER().getText()
        qualified_type_builder.set_type_qualifier(type_qualifier_name)

    def exitQualified_type(self, ctx: FhYParser.Qualified_typeContext):
        self._builder.close_qualified_type_context()

    def enterNumerical_type(self, ctx: FhYParser.Numerical_typeContext):
        self._builder.open_numerical_type_context()

    def enterDtype(self, ctx: FhYParser.DtypeContext):
        numerical_type_builder: ASTNumericalTypeBuilder = self._builder.get_current_builder()
        numerical_type_name: str = ctx.IDENTIFIER().getText()
        numerical_type_builder.set_primitive_data_type_name(numerical_type_name)

    def enterIndex_type(self, ctx: FhYParser.Index_typeContext):
        self._builder.open_index_type_context()

    def exitType(self, ctx: FhYParser.TypeContext):
        self._builder.close_type_context()

    @property
    def ast(self) -> Optional[ASTNode]:
        return self._ast


def from_parse_tree(parse_tree: FhYParser.ModuleContext) -> ASTNode:
    # TODO Jason: Add docstring
    converter = ParseTreeConverter()
    walker = ParseTreeWalker()
    walker.walk(converter, parse_tree)
    if converter.ast is None:
        # TODO Jason: Implement a better error for this ast conversion failure
        raise Exception()
    return converter.ast
