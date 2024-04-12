from typing import Optional
from antlr4 import ParseTreeWalker
from fhy.lang.parser import FhYListener, FhYParser
from fhy.lang.ast import ASTNode
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
            raise NotImplementedError()
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
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def enterFunction_arg(self, ctx: FhYParser.Function_argContext):
        self._builder.add_argument()

        argument_builder: ASTArgumentBuilder = self._builder.get_current_node()
        argument_name: str = ctx.IDENTIFIER().getText()
        argument_builder.set_name_hint(argument_name)

    def exitFunction_arg(self, ctx: FhYParser.Function_argContext):
        self._builder.close_argument_building()

    def enterQualified_type(self, ctx: FhYParser.Qualified_typeContext):
        self._builder.add_qualified_type()

        qualified_type_builder: ASTQualifiedTypeBuilder = self._builder.get_current_node()
        type_qualifier_name: str = ctx.IDENTIFIER().getText()
        qualified_type_builder.set_type_qualifier(type_qualifier_name)

    def exitQualified_type(self, ctx: FhYParser.Qualified_typeContext):
        self._builder.close_qualified_type_building()

    def enterNumerical_type(self, ctx: FhYParser.Numerical_typeContext):
        self._builder.add_numerical_type()

    def enterDtype(self, ctx: FhYParser.DtypeContext):
        numerical_type_builder: ASTNumericalTypeBuilder = self._builder.get_current_node()
        numerical_type_name: str = ctx.IDENTIFIER().getText()
        numerical_type_builder.set_primitive_data_type_name(numerical_type_name)

    def enterIndex_type(self, ctx: FhYParser.Index_typeContext):
        self._builder.add_index_type()

    def exitType(self, ctx: FhYParser.TypeContext):
        self._builder.close_type_building()

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
