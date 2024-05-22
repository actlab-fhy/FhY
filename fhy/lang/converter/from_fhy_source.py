"""FhY Source Code to AST Module Converter."""

from antlr4 import (  # type: ignore[import-untyped]
    BailErrorStrategy,
    CommonTokenStream,
    InputStream,
)
from antlr4.error.ErrorListener import ErrorListener  # type: ignore[import-untyped]

from fhy import error
from fhy.lang import ast
from fhy.lang.parser import FhYLexer, FhYParser

from .from_parse_tree import from_parse_tree


class ThrowingErrorListener(ErrorListener):
    """Custom Antlr Error Listener."""

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        message = f"Syntax error at {line}:{column} - {msg}"

        raise error.FhYSyntaxError(message) from e


# TODO: Extract out Construction of CST to another module within fhy/lang...
def create_lexer(input_str: str) -> FhYLexer:
    """Constructs the FhyLexer from Input String Source Code."""
    input_stream = InputStream(input_str)
    lexer = FhYLexer(input_stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(ThrowingErrorListener())

    return lexer


def create_parser(input_str: str) -> FhYParser:
    """Constructs the FhyParser from Input String Source Code."""
    lexer = create_lexer(input_str)
    token_stream = CommonTokenStream(lexer)
    parser = FhYParser(token_stream)
    parser._errHandler = BailErrorStrategy()
    parser.removeErrorListeners()
    parser.addErrorListener(ThrowingErrorListener())

    return parser


def _fhy_source_to_parse_tree(fhy_source_content: str) -> FhYParser.ModuleContext:
    fhy_parser = create_parser(fhy_source_content)
    tree = fhy_parser.module()

    return tree


def from_fhy_source(fhy_source_content: str) -> ast.Module:
    """Convert FhY Source Code to an AST Module."""
    tree = _fhy_source_to_parse_tree(fhy_source_content)
    _ast = from_parse_tree(tree)

    return _ast
