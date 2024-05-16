"""FhY Command Line Interface (CLI) Compiler Utility."""

import argparse
import logging
import os
import sys
from enum import IntEnum
from pathlib import Path
from typing import Optional

from fhy import __version__
from fhy.driver import CompilationOptions, Workspace, compile_fhy
from fhy.driver.ast_program_builder.source_file_ast import (
    SourceFileAST,
    build_source_file_ast,
)
from fhy.lang.pprint import pformat_ast
from fhy.lang.serialization.to_json import dump
from fhy.utils.discovery import standard_path
from fhy.utils.logger import get_logger


class Status(IntEnum):
    """Exit Codes describing the Status of CLI Request."""

    OK = 0
    FAILED = 1
    INTERRUPTED = 2
    USAGE_ERROR = 3


# TODO: We might want a filehandler to output log to a file on user flag.
# TODO: Capture Root Directory Information to define Consistent logging location
def make_logger(verbose: bool = False) -> logging.Logger:
    """Constructs a Simple Logger."""
    level: int = logging.DEBUG if verbose else logging.INFO
    log: logging.Logger = get_logger("FhY", level=level)

    return log


def arguments() -> argparse.ArgumentParser:
    """Defines FhY cli arguments."""
    parser = argparse.ArgumentParser("FhY")
    subparser = parser.add_subparsers(help="FhY subparser")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Provide more verbose logging"
    )
    parser.add_argument("--version", action="store_true", help="Display FhY Version")
    parser.add_argument(
        "-m", "--module", help="Filepath to a single module (file).", type=str
    )
    parser.add_argument(
        "--library", help="Source directory path to a project.", type=str
    )

    serialize = subparser.add_parser("serialize", help="Serialize AST Nodes")
    serialize.add_argument("-f", "--format", choices=["json", "pretty"], default=None)

    # parser.add_argument(
    #     "-o",
    #     "--output",
    #     required=False,
    #     help="Defines Output Directory. Otherwise defaults to `{filepath}_out.ast`",
    # )

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


def main() -> int:
    """Primary entry point for compilation of FhY files."""
    arg_parser: argparse.ArgumentParser = arguments()
    args: argparse.Namespace = arg_parser.parse_args()

    # Check Arguments that Shunt Compilation First
    if args.version:
        sys.stdout.write(f"FhY v{__version__}\n")
        return Status.OK

    log: logging.Logger = make_logger(args.verbose)

    if file := (args.module or args.library):
        filepath: Path = standard_path(file)
    else:
        log.error(
            "No Filepath(s) were defined. Provide a path either through "
            "`--module` or `--library` command line arguments."
        )
        return Status.USAGE_ERROR

    log.debug(f"Filepath(s) Collected: {filepath}")

    options = CompilationOptions(
        verbose=args.verbose,
    )

    if args.library:
        workspace = Workspace(root=filepath)

        try:
            compile_fhy(workspace, options)

        except KeyboardInterrupt as e:
            log.error("FhY Compilation has been Interrupted by client.", exc_info=e)
            return Status.INTERRUPTED

        except Exception as e:
            log.error("FhY Compilation has Failed", exc_info=e)
            return Status.FAILED

    elif args.module:
        result: SourceFileAST = build_source_file_ast(filepath)

        # NOTE: Currently only supporting Serialization for Single Module
        #       Roadmap should include expansion to library compilation as well
        if args.format:
            if args.format == "json":
                print(dump(result.ast, None))

            elif args.format == "pretty":
                print("\n\n")
                header = f"// {result.path}\n"
                header += "=" * len(header)
                print(header)
                print(pformat_ast(result.ast), end="\n\n")

    return Status.OK


if __name__ == "__main__":
    main()
