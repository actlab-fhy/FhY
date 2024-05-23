"""Pytest Unit Test Fixtures and Utilities.

Fixtures present in this file are innately available to all modules present within this
directory. This is not true of Subdirectories, which will need it's own conftest.py
file.

"""

from typing import Callable, List, Set

import pytest
from antlr4 import DFA, CommonTokenStream, InputStream, ParserRuleContext
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import DecisionState
from antlr4.error.ErrorListener import ErrorListener
from fhy import error as fhy_error
from fhy.lang.ast import ASTNode
from fhy.lang.converter.from_parse_tree import from_parse_tree
from fhy.lang.parser import FhYLexer, FhYParser
from fhy.logger import get_logger

log = get_logger(__name__, 10)


def get_dir(obj):
    return [i for i in dir(obj) if not i.startswith("__")]


class ThrowingErrorListener(ErrorListener):
    """An Overly Verbose, Descriptive Antlr Error Listener for Reasons."""

    def syntaxError(
        self,
        recognizer: FhYParser,
        offendingSymbol: str,
        line: int,
        column: int,
        msg: str,
        e: Exception,
    ):
        text = self.get_text(recognizer, None, None)
        context = type(recognizer._ctx).__name__
        message = f'context={context}(Line {line}:{column}) input="{offendingSymbol}" '
        # result = ParseTreeConverter().visit(recognizer._ctx)
        # message += f'text="{text}" - msg={msg} - ast={result}'
        message += f'text="{text}" - msg={msg}'

        log.error(message)

        raise fhy_error.FhYSyntaxError(message) from e

    def reportAmbiguity(
        self,
        recognizer: FhYParser,
        dfa: DFA,
        startIndex: int,
        stopIndex: int,
        exact: bool,
        ambigAlts: Set[int],
        configs: ATNConfigSet,
    ):
        report = self._report(
            recognizer,
            dfa,
            startIndex,
            stopIndex,
            configs,
        )
        report += f" exact={exact}"
        log.debug(report)

    def reportAttemptingFullContext(
        self,
        recognizer: FhYParser,
        dfa: DFA,
        startIndex: int,
        stopIndex: int,
        conflictingAlts: Set[int],
        configs: ATNConfigSet,
    ):
        report = self._report(
            recognizer,
            dfa,
            startIndex,
            stopIndex,
            configs,
        )
        log.debug(report)

    def reportContextSensitivity(
        self,
        recognizer: FhYParser,
        dfa: DFA,
        startIndex: int,
        stopIndex: int,
        prediction: int,
        configs: ATNConfigSet,
    ):
        msg = self._report(
            recognizer,
            dfa,
            startIndex,
            stopIndex,
            configs,
        )
        msg += f" predict={prediction}"
        log.debug(msg)

    def _report(
        self,
        recognizer: FhYParser,
        dfa: DFA,
        startIndex: int,
        stopIndex: int,
        configs: ATNConfigSet,
    ) -> str:
        decision = self.get_decision(recognizer, dfa)
        conflict = self.get_conflict(recognizer, configs)
        text = self.get_text(recognizer, startIndex, stopIndex)
        if (ctx := recognizer._ctx) is None:
            context = ""
        else:
            context = ctx.getText()

        message = (
            f"context={context}, d={decision}: ambigAlts={conflict}, input='{text}'"
        )

        return message

    def get_text(self, recognizer: FhYParser, start: int, stop: int) -> str:
        stream = recognizer.getTokenStream()
        text = stream.getText(start, stop)

        return text

    def get_decision(self, recognizer: FhYParser, dfa: DFA) -> str:
        decision: int = dfa.decision
        dfa.atnStartState: DecisionState
        index: int = dfa.atnStartState.ruleIndex

        rules: List[str] = recognizer.ruleNames
        if index < 0 or index >= len(rules):
            return str(decision)

        name = rules[index]
        if not name:
            return str(decision)

        return name

    def get_conflict(self, reportedAlts: Set[int], configs: ATNConfigSet) -> Set[int]:
        if reportedAlts is None:
            return reportedAlts

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
