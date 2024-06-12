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

from pygments.lexer import RegexLexer, default, words
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    Whitespace,
)


class FhYLexer(RegexLexer):
    """Lexer for the FhY language."""

    name = "FhY"
    aliases = ["fhy"]
    filenames = ["*.fhy"]

    whitespace = (r"\s+", Whitespace)

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
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
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

    tokens = {
        "root": [
            # Newline
            (r"\n", Whitespace),
            # Whitespace
            whitespace,
            # Comments
            (r"//.*?$", Comment.Single),
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
            (
                r"[0-9][0-9_]*(\.[0-9_]*)?[jJ]([eE][+\-]?[0-9][0-9_]*)?",
                Number.Complex
            ),
            (r"[0-9][0-9_]*", Number.Integer),
            # Operators and Punctuation
            (r"[{}()\[\],.;]|(->)", Punctuation),
            (r"[+\-*/%&|<>^!~=:?]", Operator),
            # Identifiers
            (identifier_pattern, Name),
        ],
        "function_name": [
            whitespace,
            (identifier_pattern, Name.Function, "#pop"),
            default("#pop"),
        ],
    }


def setup(app):
    """Setup Code Lexer."""
    app.add_lexer("FhY", FhYLexer)
