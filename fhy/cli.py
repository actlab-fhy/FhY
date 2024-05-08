"""FhY Command Line Interface (CLI) Compiler Utility."""

import argparse
import logging
import os
from contextlib import contextmanager
from enum import IntEnum
from typing import List, Optional

from fhy.lang.ast import Module
from fhy.lang.ast_builder import from_fhy_source
from fhy.lang.pprint import pformat_ast
from fhy.lang.serialization.to_json import dump
from fhy.utils import error
from fhy.utils.discovery import confirm_files
from fhy.utils.logger import get_logger


class Status(IntEnum):
    """Exit Codes describing the Status of CLI Request."""

    OK = 0
    FAILED = 1
    INTERRUPTED = 2
    USAGE_ERROR = 3


# TODO: We might want a filehandler to output log to a file on user flag.
def make_logger(verbose: bool = False) -> logging.Logger:
    """Constructs a Simple Logger."""
    level: int = logging.DEBUG if verbose else logging.INFO
    log: logging.Logger = get_logger("FhY", level=level)
    return log


def arguments() -> argparse.ArgumentParser:
    """Defines FhY cli arguments."""
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
        "-j", "--json", action="store_true", help="Serialize AST Nodes to JSON Format"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="Defines Output Directory. Otherwise defaults to `{filepath}_out.ast`",
    )

    return parser


def output_path(filepath: str, override_dir: Optional[str]) -> str:
    """Construct an output path for a given file."""
    if override_dir is None:
        parent_relative = os.path.join(filepath, os.pardir)
        parent_dir = os.path.abspath(parent_relative)
    else:
        if not os.path.exists(override_dir):
            raise FileNotFoundError(
                f"Output Filepath Directory Not Found: {override_dir}"
            )
        parent_dir = os.path.abspath(override_dir)

    basename = os.path.basename(filepath).split(".")[0]
    new_path = os.path.join(parent_dir, f"{basename}_out.ast")

    return new_path


@contextmanager
def open_file(filepath: str, mode: str):
    """Context manager to Open a File, with Better Error Reporting and Handling."""
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
    """Primary Entry Point for Compilation of FhY Files."""
    arg_parser = arguments()
    args = arg_parser.parse_args()

    filepaths: List[str] = []
    for f in args.files:
        filepaths.extend(confirm_files(f))

    log = make_logger(args.verbose)
    log.debug(f"Filepaths Collected: {filepaths}")

    if args.output is not None:
        raise NotImplementedError(
            "We are not supporting Custom Output Directories yet."
        )

    # Quick Double Check before Proceeding
    if len(filepaths) == 0:
        log.error(
            f"No Filepaths were collected from provided file arguments: {args.files}"
        )
        raise error.UsageError("No Filepaths were collected, or defined to compile.")

    # NOTE: This should go into a `Root` or `Project` ASTNode
    constructed: List[Module] = []
    for file in filepaths:
        log.debug(f"Started Compiling: {file}")

        # Out of an abundance of caution, Fail Fast during reading.
        with open_file(file, "r") as (text, err):
            if err is not None:
                raise FileExistsError(f"Unable to Read provided file: {file}") from err
            if text is None:
                raise FileExistsError(f"No Text was read from file: {file}")

        ast = from_fhy_source(text)
        if ast is None or not isinstance(ast, Module):
            log.error(f"Unable to Construct AST from: {file}")
            raise error.FhYASTBuildError(f"Unable to Construct AST from: {file}")
        constructed.append(ast)

    # TODO: Rename this, since we are essentially returning our ASTNodes to FhY Text
    #       Pretty Printing should be a text representation of our nodes (and not json)
    if args.pretty:
        print("\n\n")
        for fname, node in zip(filepaths, constructed):
            header = f"// {fname}\n"
            header += "=" * len(header)
            print(header)
            print(pformat_ast(node), end="\n\n")

    if args.json:
        print([dump(node, None) for node in constructed])

    return Status.OK


if __name__ == "__main__":
    main()
