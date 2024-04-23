import argparse
import logging
import os
from contextlib import contextmanager
from typing import List, Optional, Tuple

from antlr4 import (
    BailErrorStrategy,
    CommonTokenStream,
    InputStream,
)
from antlr4.error.ErrorListener import ErrorListener

from fhy.lang.ast import ASTNode, Module
from fhy.lang.ast_builder import from_parse_tree
from fhy.lang.parser import FhYLexer, FhYParser
from fhy.lang.printer import pprint_ast
from fhy.utils.logger import get_logger


class ThrowingErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise SyntaxError(f"Syntax error at {line}:{column} - {msg}")


def make_logger(verbose: bool = False) -> logging.Logger:
    """Constructs a Simple Logger"""
    level: int = logging.DEBUG if verbose else logging.INFO
    log: logging.Logger = get_logger("FhY", level=level)
    return log


def collect_files(directory: str, endswith: Optional[str] = None) -> List[str]:
    """Recursively Collects Files from a Directory or a given file type (if provided)"""
    files: List[str] = []
    for f in os.scandir(directory):
        if f.is_dir():
            # NOTE: We may have Rules about Subfolders. e.g.
            # root = os.path.join(f.path, "__root__.fhy")
            # if os.path.exists(root): ...
            files.extend(collect_files(f.path))
            continue

        elif f.is_file() and f.path != ".DS_Store":
            if endswith is None or f.path.endswith(endswith):
                files.append(f.path)

        else:
            raise ValueError("")

    return files


def confirm_files(filepath: str) -> List[str]:
    path = os.path.abspath(filepath)
    if not os.path.exists(path):
        raise FileExistsError(f"File does not Exist: {path}")
    elif os.path.isdir(path):
        return collect_files(path)
    elif os.path.isfile(path) and path.endswith(".fhy"):
        return [path]
    else:
        raise FileNotFoundError(f"Invalid Filetype Provided: {path}")


def arguments() -> argparse.ArgumentParser:
    """Defines FhY cli arguments"""
    parser = argparse.ArgumentParser("FhY")
    parser.add_argument(
        "files",
        nargs="+",
        help="Filepath(s) or Directory(ies) of FhY files to compile.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Provide more verbose logging"
    )
    parser.add_argument(
        "-p", "--pretty", action="store_true", help="Pretty Print Constructed AST Nodes"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="Defines Output Directory. Otherwise defaults to `{filepath}_out.ast`",
    )

    return parser


def create_lexer(input_str: str) -> FhYLexer:
    """Constructs the FhyLexer from an Input String Source Code"""
    input_stream = InputStream(input_str)
    lexer = FhYLexer(input_stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(ThrowingErrorListener())
    return lexer


def create_parser(input_str: str) -> FhYParser:
    """Constructs the FhyParser from an Input String Source Code"""
    lexer = create_lexer(input_str)
    token_stream = CommonTokenStream(lexer)
    parser = FhYParser(token_stream)
    parser._errHandler = BailErrorStrategy()
    parser.removeErrorListeners()
    parser.addErrorListener(ThrowingErrorListener())

    return parser


def construct_ast(input_str: str) -> ASTNode:
    fhy_parser = create_parser(input_str)
    tree = fhy_parser.module()
    _ast = from_parse_tree(tree)

    return _ast


def default_output_path(filepath: str) -> str:
    """Construct a default output path for a given file"""
    parent_relative = os.path.join(filepath, os.pardir)
    parent_dir = os.path.abspath(parent_relative)
    basename = os.path.basename(filepath).split(".")[0]

    new_path = os.path.join(parent_dir, f"{basename}_out.ast")
    return new_path


@contextmanager
def open_file(filepath: str, mode: str):
    """Context manager to Open a File, with Better Error Reporting and Handling"""
    try:
        stream = open(filepath, mode)
    except Exception as e:
        yield None, e
    else:
        try:
            yield stream.read(), None
        finally:
            stream.close()


def main():
    """Primary Entry Point for Compilation of FhY Files"""
    arg_parser = arguments()
    args = arg_parser.parse_args()

    filepaths: List[str] = []
    for f in args.files:
        filepaths.extend(confirm_files(f))

    log = make_logger(args.verbose)
    log.debug(f"Filepaths Collected: {filepaths}")

    if args.output is not None:
        raise NotImplementedError("We are not supporting Cutom Ouptut Directories yet.")

    # Quick Double Check before Proceeding
    if len(filepaths) == 0:
        log.error(
            f"No Filepaths were collected from provided file arguments: {args.files}"
        )
        raise Exception("")

    # This should go into a `Root` or `Project` ASTNode
    constructed: List[Module] = []
    for file in filepaths:
        log.debug(f"Started Compiling: {file}")

        # Out of an abundance of caution, raise Early during reading.
        with open_file(file, "r") as (text, err):
            if err is not None:
                raise err
            if text is None:
                raise Exception(f"No Text was read from file: {file}")

        ast = construct_ast(text)
        if ast is None or not isinstance(ast, Module):
            log.error(f"Unable to Construct AST from: {file}")
            raise Exception("Unable to Construct AST")
        constructed.append(ast)

    if args.pretty:
        print("\n\n")
        for fname, node in zip(filepaths, constructed):
            header = f"// {fname}"
            header += "\n" + "=" * len(header)
            print(header)
            print(pprint_ast(node), end="\n\n")

    # TODO: Serialize the AST Nodes
    return constructed


if __name__ == "__main__":
    main()
