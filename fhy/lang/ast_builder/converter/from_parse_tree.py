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
            self._builder.add_operation(function_name)
        else:
            raise NotImplementedError()

    def enterFunction_arg(self, ctx: FhYParser.Function_argContext):
        argument_name: str = ctx.IDENTIFIER().getText()
        self._builder.add_argument(argument_name)

    def exitFunction_arg(self, ctx: FhYParser.Function_argContext):
        self._builder.close_argument_building()

    def enterQualified_type(self, ctx: FhYParser.Qualified_typeContext):
        type_qualifier_name: str = ctx.IDENTIFIER().getText()
        self._builder.add_qualified_type(type_qualifier_name)

    def exitQualified_type(self, ctx: FhYParser.Qualified_typeContext):
        self._builder.close_qualified_type_building()

    def enterShape(self, ctx: FhYParser.ShapeContext):
        # TODO: This is a little Wonky, and Won't support Expressions.
        shapes: List[str] = ctx.getText().split(",")
        self._builder.add_shape(shapes)

    # def enterAtom(self, ctx: FhYParser.AtomContext):
    #     # Will likely need this instead of enterShape
    #     return super().enterAtom(ctx)

    def enterNumerical_type(self, ctx: FhYParser.Numerical_typeContext):
        self._builder.add_numerical_type()

    def enterDtype(self, ctx: FhYParser.DtypeContext):
        numerical_type_name: str = ctx.IDENTIFIER().getText()
        self._builder.add_dtype(numerical_type_name)

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
    assert len(converter._builder._node_stack) == 0, "Incomplete AST Build."
    if converter.ast is None:
        # TODO Jason: Implement a better error for this ast conversion failure
        raise Exception()

    return converter.ast
