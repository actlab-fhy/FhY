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

"""Pygments lexer for the FhY language."""

from collections.abc import Iterator
from typing import Iterable
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import RegexLexer, bygroups, default, include, words
from pygments.token import (
    _TokenType,
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    Whitespace,
)


def generate_identity_callback(token_type, action):
    def callback(lexer, match):
        action(lexer, match)
        yield match.start(), token_type, match.group(0)

    return callback


class FhYLexer(RegexLexer):
    """Lexer for the FhY language."""

    name = "FhY"
    aliases = ["fhy"]
    filenames = ["*.fhy"]

    BUILTIN_FUNCTIONS = {
        "sum",
        "prod",
        "min",
        "max",
    }

    whitespace = (r"\s+", Whitespace)

    keyword_import_pattern = words(
        (
            "import",
            "from",
        ),
        suffix=r"\b",
    )

    keyword_function_pattern = words(
        (
            "proc",
            "op",
            "native",
        ),
        suffix=r"\b",
    )

    keyword_type_qualifier_pattern = words(
        (
            "input",
            "output",
            "state",
            "param",
            "temp",
        ),
        suffix=r"\b",
    )

    keyword_type_pattern = words(
        (
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
            "complex32",
            "complex64",
            "complex128",
            "index",
        ),
        suffix=r"\b",
    )

    keyword_other_pattern = words(
        (
            "forall",
            "if",
            "else",
            "return",
        ),
        suffix=r"\b",
    )

    identifier_pattern = r"[a-zA-Z_]\w*"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._generic_types = set()
        self._params = set()

    def reset(self):
        self.reset_generic_types()
        self.reset_params()

    def reset_generic_types(self):
        self._generic_types.clear()

    def reset_params(self):
        self._params.clear()

    def add_generic_type(self, type_name):
        self._generic_types.add(type_name)

    def add_param(self, param_name):
        self._params.add(param_name)

    def get_tokens_unprocessed(self, text):
        for index, token, value in RegexLexer.get_tokens_unprocessed(self, text):
            if token is Name and value in self.BUILTIN_FUNCTIONS:
                yield index, Name.Builtin, value
            elif token is Name and value in self._generic_types:
                yield index, Keyword.Pseudo, value
            elif token is Name and value in self._params:
                yield index, Name.Constant, value
            else:
                yield index, token, value

    tokens = {
        "root": [
            # Newline
            (r"\n", Whitespace),
            # Whitespace
            whitespace,
            # Comments
            (r"#.*?$", Comment.Single),
            # Imports
            (keyword_import_pattern, Keyword.Namespace),
            # Functions
            (keyword_function_pattern, Keyword, "function_name"),
            # Type Qualifiers
            (keyword_type_qualifier_pattern, Keyword.Reserved),
            # Types
            (keyword_type_pattern, Keyword.Type),
            # Other Keywords
            (keyword_other_pattern, Keyword),
            # Binary Literals
            (r"0b[01_]+", Number.Bin),
            # Octal Literals
            (r"0o[0-7_]+", Number.Oct),
            # Hexadecimal Literals
            (r"0[xX][0-9a-fA-F_]+", Number.Hex),
            # Decimal Literals
            (
                r"[0-9][0-9_]*(\.[0-9_]+[eE][+\-]?[0-9_]+|"
                r"\.[0-9_]*(?!\.)|[eE][+\-]?[0-9_]+)",
                Number.Float,
            ),
            # Complex Literals
            (r"[0-9][0-9_]*(\.[0-9_]*)?[jJ]([eE][+\-]?[0-9][0-9_]*)?", Number.Complex),
            (r"[0-9][0-9_]*", Number.Integer),
            # Operators and Punctuation
            (r"[{}()\[\],.;]|(->)", Punctuation),
            (r"[+\-*/%&|<>^!~=:?]", Operator),
            # Identifiers
            (identifier_pattern, Name),
        ],
        "function_name": [
            # Function name rule following a function keyword. Only whitespace
            # or an identifier are allowed after the function keyword
            whitespace,
            (
                identifier_pattern,
                generate_identity_callback(Name.Function, lambda l, m: l.reset()),
                "function_signature",
            ),
        ],
        "function_signature": [
            # Function signature rule following a function name. The function
            # signature is the part of the function definition that includes
            # the function arguments, return type, and generic template types.
            # Each rule here either transitions to a new state or pops the
            # current state off the stack if the signature is complete.
            whitespace,
            (r"\(", Punctuation, "function_args"),
            (r"(->)", Punctuation, "function_return_type"),
            (r"<", Punctuation, "function_generics"),
            (r"[{;]", Punctuation, "#pop:2"),
        ],
        "function_args": [
            # Function arguments rule following an open parenthesis. The rule
            # is the same as the root except for when opening the shape of a
            # type or when closing the argument list.
            (r"\[", Punctuation, "function_arg_shape"),
            (r"\)", Punctuation, "#pop"),
            include("root"),
        ],
        "function_return_type": [
            # Function return type rule following a function arrow. The rule
            # is the same as the root except for when closing the return type.
            (r"(;{)", Punctuation, "#pop"),
            include("root"),
        ],
        "function_generics": [
            # Function generics rule following an open angle bracket. The rule
            # only allows for whitespace, commas, and identifiers. The rule
            # transitions back to the function signature state when the closing
            # angle bracket is encountered.
            # The identifier pattern rule is used to store the generic types
            # for the function.
            whitespace,
            (
                identifier_pattern,
                generate_identity_callback(
                    Keyword.Pseudo, lambda l, m: l.add_generic_type(m.group(0))
                ),
            ),
            (r",", Punctuation),
            (r">", Punctuation, "#pop"),
        ],
        "function_arg_shape": [
            # Function argument shape rule following an open square bracket.
            # The rule is the same as the root except for when closing the shape
            # and when encountering an identifier, which must be a parameter and
            # therefore is a constant throughout the rest of the function and
            # is highlighted as such.
            (
                identifier_pattern,
                generate_identity_callback(
                    Name.Constant, lambda l, m: l.add_param(m.group(0))
                ),
            ),
            (r"\]", Punctuation, "#pop"),
            include("root"),
        ],
    }


def _setup_context(app, pagename, templatename, context, doctree):
    """Setup context for the HTML page."""
    context["pygments_highlight_fhy"] = lambda code: highlight(code, FhYLexer(), HtmlFormatter())


def setup(app):
    """Setup code lexer."""
    app.add_lexer("FhY", FhYLexer)
    app.connect("html-page-context", _setup_context)
