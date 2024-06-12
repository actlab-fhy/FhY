# Copyright (c) 2024 FhY Developers
# Christopher Priebe <cpriebe@ucsd.edu>
# Jason C Del Rio <j3delrio@ucsd.edu>
# Hadi S Esmaeilzadeh <hadi@ucsd.edu>
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be
# used to endorse or promote products derived from this software without specific prior
# written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""FhY source code to AST module converter."""

import logging

from antlr4 import (  # type: ignore[import-untyped]  # type: ignore[import-untyped]
    DFA,
    CommonTokenStream,
    InputStream,
)
from antlr4.atn.ATNConfigSet import ATNConfigSet  # type: ignore[import-untyped]
from antlr4.error.ErrorListener import (  # type: ignore[import-untyped]
    ErrorListener,
)

from fhy import error
from fhy.lang import ast
from fhy.lang.parser import FhYLexer, FhYParser

from .from_parse_tree import from_parse_tree

_log = logging.getLogger(__name__)


class ThrowingErrorListener(ErrorListener):
    """An Overly Verbose, Descriptive Antlr Error Listener for Reasons."""

    log: logging.Logger

    def __init__(self, log: logging.Logger = _log) -> None:
        super().__init__()
        self.log = log

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

        self.log.error(message)

        raise error.FhYSyntaxError(message) from e

    def reportAmbiguity(
        self,
        recognizer: FhYParser,
        dfa: DFA,
        startIndex: int,
        stopIndex: int,
        exact: bool,
        ambigAlts: set[int],
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
        self.log.debug(report)

    def reportAttemptingFullContext(
        self,
        recognizer: FhYParser,
        dfa: DFA,
        startIndex: int,
        stopIndex: int,
        conflictingAlts: set[int],
        configs: ATNConfigSet,
    ):
        report = self._report(
            recognizer,
            dfa,
            startIndex,
            stopIndex,
            configs,
        )
        self.log.debug(report)

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
        self.log.debug(msg)

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

    def get_text(
        self,
        recognizer: FhYParser,
        start: int | None = None,
        stop: int | None = None,
    ) -> str:
        stream = recognizer.getTokenStream()
        text = stream.getText(start, stop)

        return text

    def get_decision(self, recognizer: FhYParser, dfa: DFA) -> str:
        decision: int = dfa.decision
        index: int = dfa.atnStartState.ruleIndex

        rules: list[str] = recognizer.ruleNames
        if index < 0 or index >= len(rules):
            return str(decision)

        name = rules[index]
        if not name:
            return str(decision)

        return name

    def get_conflict(self, reportedAlts: set[int], configs: ATNConfigSet) -> set[int]:
        if reportedAlts is None:
            return reportedAlts

        return {i.alt for i in configs}


def create_lexer(input_str: str, log: logging.Logger = _log) -> FhYLexer:
    """Construct the FhyLexer from input string source code."""
    input_stream = InputStream(input_str)
    lexer = FhYLexer(input_stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(ThrowingErrorListener(log))

    return lexer


def create_parser(input_str: str, log: logging.Logger = _log) -> FhYParser:
    """Construct the FhyParser from input string source code."""
    lexer = create_lexer(input_str, log)
    token_stream = CommonTokenStream(lexer)
    parser = FhYParser(token_stream)
    # parser._errHandler = BailErrorStrategy()
    parser.removeErrorListeners()
    parser.addErrorListener(ThrowingErrorListener(log))

    return parser


def _fhy_source_to_parse_tree(
    fhy_source_content: str, log: logging.Logger = _log
) -> FhYParser.ModuleContext:
    fhy_parser = create_parser(fhy_source_content, log)
    tree = fhy_parser.module()

    return tree


def from_fhy_source(
    fhy_source_content: str,
    source: ast.Source | None = None,
    log: logging.Logger = _log,
) -> ast.Module:
    """Convert FhY source code into corresponding AST module representation.

    Args:
        fhy_source_content (str): FhY source code text.
        source (optional, Source): Define code module source path or namespace.
        log (logging.Logger): Inject a logger to control debugging information during
            parsing.

    Returns:
        (ast.Module): AST module representation of input source code.

    """
    tree = _fhy_source_to_parse_tree(fhy_source_content, log)
    _ast = from_parse_tree(tree, source)

    return _ast
