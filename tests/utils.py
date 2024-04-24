""" """

from typing import Any, Callable, List

import pytest
from antlr4 import BailErrorStrategy, CommonTokenStream, InputStream, ParserRuleContext
from antlr4.error.ErrorListener import ErrorListener

from fhy.lang.ast import ASTNode
from fhy.lang.ast_builder import from_parse_tree
from fhy.lang.parser import FhYLexer, FhYParser


class ThrowingErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise SyntaxError(f"Syntax error at {line}:{column} - {msg}")


@pytest.fixture(scope="module")
def lexer() -> Callable[[str], FhYLexer]:
    def create_lexer(input_str: str):
        input_stream = InputStream(input_str)
        lexer = FhYLexer(input_stream)
        lexer.removeErrorListeners()
        lexer.addErrorListener(ThrowingErrorListener())
        return lexer

    return create_lexer


@pytest.fixture(scope="module")
def parser(lexer):
    def create_parser(input_str):
        lexer_instance = lexer(input_str)
        token_stream = CommonTokenStream(lexer_instance)
        parser = FhYParser(token_stream)
        parser._errHandler = BailErrorStrategy()
        parser.removeErrorListeners()
        parser.addErrorListener(ThrowingErrorListener())
        return parser

    return create_parser


def parse_file_contents(parser, file_contents: str) -> ParserRuleContext:
    parse_tree = parser(file_contents).module()
    assert parse_tree is not None, "Expected parse tree for module"
    return parse_tree


def construct_ast(parser, file_contents: str) -> ASTNode:
    """Create a Concrete Syntax (Parse) Tree for file source, and convert to AST
    representation.

    """
    cst = parse_file_contents(parser, file_contents)
    return from_parse_tree(cst)


def list_to_types(xs: List[Any]) -> List[type]:
    return [type(x) for x in xs]
