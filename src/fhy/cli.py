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

"""FhY Command Line Interface (CLI) Compiler Utility."""

import argparse
import logging
import sys
from enum import IntEnum
from pathlib import Path

from fhy import __version__
from fhy.driver import CompilationOptions, Workspace, compile_fhy
from fhy.driver.ast_program_builder.source_file_ast import (
    SourceFileAST,
    build_source_file_ast,
)
from fhy.driver.file_reader import standard_path
from fhy.lang.ast.pprint import pformat_ast
from fhy.lang.ast.serialization.to_json import dump
from fhy.logger import add_file_handler, get_logger


class Status(IntEnum):
    """Exit Codes describing the Status of CLI Request."""

    OK = 0
    FAILED = 1
    INTERRUPTED = 2
    USAGE_ERROR = 3


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
    parser.add_argument("--log-file", help="filepath to stream (save) log messages to.")
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
    #     help="Defines Output Directory. Otherwise defaults to \"{filepath}_out.ast\"",
    # )

    return parser


# TODO: Define More Exacting Rules on Expectations of user override.
#       Since we plan on creating a temporary directory, this will be on hold.
# def output_path(filepath: str, override_dir: Optional[str]) -> str:
#     """Construct an output path for a given file."""
#     if override_dir is None:
#         parent_relative = os.path.join(filepath, os.pardir)
#         parent_dir = os.path.abspath(parent_relative)

#     else:
#         if not os.path.exists(override_dir):
#             raise FileNotFoundError(
#                 f"Output Filepath Directory Not Found: {override_dir}"
#             )
#         parent_dir = os.path.abspath(override_dir)

#     basename = os.path.basename(filepath).split(".")[0]
#     new_path = os.path.join(parent_dir, f"{basename}_out.ast")

#     return new_path


def main() -> int:  # noqa: PLR0912
    """Primary entry point for compilation of FhY files."""
    arg_parser: argparse.ArgumentParser = arguments()
    args: argparse.Namespace = arg_parser.parse_args()

    # Check Arguments that Shunt Compilation First
    if args.version:
        sys.stdout.write(f"FhY v{__version__}\n")
        return Status.OK

    log: logging.Logger = make_logger(args.verbose)

    # TODO: If not specified, we should have a default log location within private
    #       directory. Or in addition to client handler.
    if args.log_file:
        add_file_handler(log, args.log_file)
        log.debug("Added File Handler to log instance: %s", args.log_file)

    if file := (args.module or args.library):
        try:
            filepath: Path = standard_path(file)

        except (TypeError, FileExistsError) as e:
            log.error(str(e), exc_info=e)
            return Status.USAGE_ERROR

    else:
        log.error(
            "No Filepath(s) were defined. Provide a path either through "
            '"--module" or "--library" command line arguments.'
        )
        return Status.USAGE_ERROR

    log.debug(f"Filepath(s) Collected: {filepath}")

    options = CompilationOptions(
        verbose=args.verbose,
    )

    if args.library:
        workspace = Workspace(root=filepath)

        try:
            compile_fhy(workspace, options, log)

        except KeyboardInterrupt as e:
            log.error("FhY Compilation has been Interrupted by client.", exc_info=e)
            return Status.INTERRUPTED

        except Exception as e:
            log.error("FhY Compilation has Failed.", exc_info=e)
            return Status.FAILED

    elif args.module:
        result: SourceFileAST = build_source_file_ast(filepath)

        # NOTE: Currently only supporting Serialization for Single Module
        #       Roadmap should include expansion to library compilation as well
        if args.format:
            if args.format == "json":
                sys.stdout.write(dump(result.ast, None))
                sys.stdout.write("\n")

            elif args.format == "pretty":
                header = f"// {result.path}\n"
                header += "=" * (len(header) - 1)
                sys.stdout.write(f"\n\n{header}\n")
                sys.stdout.write(f"{pformat_ast(result.ast)}\n\n")

    return Status.OK


if __name__ == "__main__":
    main()
