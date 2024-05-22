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
    """Custom Antlr error listener."""

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        message = f"Syntax error at {line}:{column} - {msg}"

        raise error.FhYSyntaxError(message) from e


# TODO: Extract out Construction of CST to another module within fhy/lang...
def create_lexer(input_str: str) -> FhYLexer:
    """Constructs the FhyLexer from input string source code."""
    input_stream = InputStream(input_str)
    lexer = FhYLexer(input_stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(ThrowingErrorListener())

    return lexer


def create_parser(input_str: str) -> FhYParser:
    """Constructs the FhyParser from input string source code."""
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
    """Convert FhY source code to an AST module."""
    tree = _fhy_source_to_parse_tree(fhy_source_content)
    _ast = from_parse_tree(tree)

    return _ast
