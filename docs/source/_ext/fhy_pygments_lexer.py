""" Pygments lexer for the FhY language.
"""

from pygments.lexer import RegexLexer, include, bygroups, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
    Number, Punctuation, Whitespace


class FhYLexer(RegexLexer):
    """Lexer for the FhY language.
    """
    name = "FhY"
    aliases = ["fhy"]
    filenames = ["*.fhy"]

    whitespace = (r"\s+", Whitespace)

    keyword_function_pattern = words((
        "proc", "op", "native",
    ), suffix=r"\b")

    keyword_type_qualifier_pattern = words((
        "input", "output", "state", "param", "temp",
    ), suffix=r"\b")

    keyword_type_pattern = words((
        "int32", "float32", "index",
    ), suffix=r"\b")

    keyword_other_pattern = words((
        "forall", "if", "else", "return",
    ), suffix=r"\b")

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
            (r"[0-9][0-9_]*(\.[0-9_]+[eE][+\-]?[0-9_]+|"
             r"\.[0-9_]*(?!\.)|[eE][+\-]?[0-9_]+)", Number.Float),
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
    app.add_lexer("FhY", FhYLexer)
