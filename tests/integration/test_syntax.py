"""Tests the conversion from ANTLR parse tree to FhY AST and the AST printer"""
import pytest

from typing import Any, Callable, List, Optional, Type
from antlr4 import (
    BailErrorStrategy,
    CommonTokenStream,
    InputStream,
    ParserRuleContext,
    RecognitionException,
)
from antlr4.error.ErrorListener import ErrorListener

from fhy import ir
from fhy.lang import ast
from fhy.lang.ast_builder import from_parse_tree
from fhy.lang.parser import FhYLexer, FhYParser
from fhy.lang.printer import pprint_ast


# TODO: the parsing code is used in two files currently; centralize it in a utils file
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


def _parse_file_contents(parser, file_contents: str) -> ParserRuleContext:
    parse_tree = parser(file_contents).module()
    assert parse_tree is not None, "Expected parse tree for module"
    return parse_tree


def test_matmul_proc(parser):
    source_file_content = """
proc matmul(input int32[m, n] A, input int32[n, p] B, output int32[m, p] C) {
   temp index[1:m] i;
   temp index[1:p] j;
   temp index[1:n] k;
   C[i, j] = sum[k](A[i, k] * B[k, j]);
}
"""
    parse_tree = _parse_file_contents(parser, source_file_content)

    _ast = from_parse_tree(parse_tree)
    pprinted_ast = pprint_ast(_ast, indent_char="  ")

    expected_pprinted_ast = """proc matmul(input int32[m, n] A, input int32[n, p] B, output int32[m, p] C) {
  temp index[1:m:1] i;
  temp index[1:p:1] j;
  temp index[1:n:1] k;
  C[i, j] = sum<>[k]((A[i, k] * B[k, j]));
}"""
    assert pprinted_ast == expected_pprinted_ast, f"Expected:\n{expected_pprinted_ast}\nGot:\n{pprinted_ast}"
