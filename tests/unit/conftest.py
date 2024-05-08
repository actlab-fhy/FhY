"""Pytest Unit Test Fixtures and Utilities.

Fixtures present in this file are innately available to all modules present within this
directory. This is not true of Subdirectories, which will need it's own conftest.py
file.

"""

from typing import Callable, Set, Tuple

import pytest
from antlr4 import CommonTokenStream, InputStream, ParserRuleContext
from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.ErrorStrategy import ParseCancellationException

from fhy.lang.ast import ASTNode
from fhy.lang.ast_builder.converter.from_parse_tree import from_parse_tree
from fhy.lang.parser import FhYLexer, FhYParser
from fhy.utils import error as fhy_error
from fhy.utils.logger import get_logger

log = get_logger(__name__)


def get_dir(obj):
    return [i for i in dir(obj) if not i.startswith("__")]


class ThrowingErrorListener(ErrorListener):
    """An Overly Verbose, Descriptive Antlr Error Listener for Reasons."""

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        log.info((recognizer, offendingSymbol, line, column, msg, e))

        raise fhy_error.FhYSyntaxError(
            f"Syntax error at {line}:{column} - {msg}"
        ) from e

    def reportAmbiguity(
        self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs
    ):
        msg = (recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs)
        log.info(msg)
        message = f"Ambiguity error at {startIndex}:{stopIndex}"

        raise ParseCancellationException(f"reportAmbiguity: {message}")

    def reportAttemptingFullContext(
        self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs
    ):
        msg = (recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs)
        log.info(msg)

        decision = self.get_decision(recognizer, dfa)
        conflict = self.get_conflict(recognizer, configs)
        text = self.get_text(recognizer, startIndex, stopIndex)
        context = recognizer._ctx
        message = (
            "reportAttemptingFullContext: context={}, d={}: ambigAlts={}, input='{}'"
        )

        raise ParseCancellationException(
            message.format(context.getText(), decision, conflict, text)
        )

    def reportContextSensitivity(
        self, recognizer, dfa, startIndex, stopIndex, prediction, configs
    ):
        log.info((recognizer, dfa, startIndex, stopIndex, prediction, configs))
        message = f"ContextSensitivity error at {startIndex}:{stopIndex}"

        raise ParseCancellationException(f"reportContextSensitivity: {message}")

    def get_text(self, recognizer, start: int, stop: int) -> str:
        stream = recognizer.getTokenStream()
        text = stream.getText(start, stop)

        return text

    def get_decision(self, recognizer, dfa) -> Tuple[int, int]:
        decision: int = dfa.decision
        rule_index: int = dfa.atnStartState.ruleIndex
        #  Python doesn't have this method to map Index values to Names
        # rule_names: List[str] = recognizer.getRuleNames()

        return decision, rule_index

    def get_conflict(self, reportedAlts, configs) -> Set[int]:
        if reportedAlts is None:
            return reportedAlts

        # Use a set to return unique values
        return {i.alt for i in configs}


@pytest.fixture(scope="module")
def lexer() -> Callable[[str], FhYLexer]:
    def create_lexer(input_str: str) -> FhYLexer:
        input_stream = InputStream(input_str)
        lexer = FhYLexer(input_stream)
        lexer.removeErrorListeners()
        lexer.addErrorListener(ThrowingErrorListener())

        return lexer

    return create_lexer


@pytest.fixture(scope="module")
def parser(lexer) -> Callable[[str], FhYParser]:
    def create_parser(input_str) -> FhYParser:
        lexer_instance = lexer(input_str)
        token_stream = CommonTokenStream(lexer_instance)
        parser = FhYParser(token_stream)
        # parser._errHandler = BailErrorStrategy()
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
