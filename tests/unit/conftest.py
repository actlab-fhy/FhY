"""Pytest Unit Test Fixtures and Utilities.

Fixtures present in this file are innately available to all modules present within this
directory. This is not true of Subdirectories, which will need it's own conftest.py
file.

"""

from typing import Any, Callable, List

import pytest
from antlr4 import BailErrorStrategy, CommonTokenStream, InputStream, ParserRuleContext
from antlr4.error.ErrorListener import ErrorListener

from fhy.lang.ast import ASTNode
from fhy.lang.ast_builder import from_parse_tree
from fhy.lang.parser import FhYLexer, FhYParser


class ThrowingErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise SyntaxError(f"Syntax error at {line}:{column} - {msg}") from e


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


@pytest.fixture
def parse_file_contents(parser) -> Callable[[str], ParserRuleContext]:
    """Build a Concrete Syntax Tree from Raw Source Text (file) using Antlr."""

    def _inner(source: str):
        parse_tree = parser(source).module()
        assert parse_tree is not None, "Expected parse tree for module"
        return parse_tree

    return _inner


@pytest.fixture
def construct_ast(parse_file_contents) -> Callable[[str], ASTNode]:
    """Construct an Abstract Syntax Tree (AST) from a raw text file source."""

    def _inner(source: str) -> ASTNode:
        cst = parse_file_contents(source)
        ast = from_parse_tree(cst)
        return ast

    return _inner
