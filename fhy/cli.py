"""FhY Command Line Interface (CLI) Compiler Utility."""

import argparse
import logging
import os
from enum import IntEnum
from pathlib import Path
from typing import List, Optional

from fhy.driver import CompilationOptions, Workspace, compile_fhy
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
# TODO: Capture Root Directory Information to define Consistent logging location
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
        # return Status.USAGE_ERROR
        raise error.UsageError("No Filepaths were collected, or defined to compile.")

    options = CompilationOptions(
        verbose=args.verbose,
    )

    workspace = Workspace(root=Path(filepaths[0]))

    # TODO: have compilation function return status
    # NOTE: I Disagree Here. compile_fhy should not be client facing. Main should.

    try:
        compile_fhy(workspace, options)

    except KeyboardInterrupt as e:
        log.error("FhY Compilation has been Interrupted by client.", exc_info=e)
        return Status.INTERRUPTED

    except Exception as e:
        log.error("FhY Compilation has Failed", exc_info=e)
        return Status.FAILED

    # NOTE: This should go into a `Root` or `Project` ASTNode
    # constructed: List[Module] = []
    # for file in filepaths:
    #     log.debug(f"Started Compiling: {file}")

    #     # Out of an abundance of caution, Fail Fast during reading.

    #     if ast is None or not isinstance(ast, Module):
    #         log.error(f"Unable to Construct AST from: {file}")
    #         raise error.FhYASTBuildError(f"Unable to Construct AST from: {file}")
    #     constructed.append(ast)

    # # TODO: Rename this, since we are essentially returning our ASTNodes to FhY Text
    # #       Pretty Printing should be a text repr of our nodes (and not json)
    # if args.pretty:
    #     print("\n\n")
    #     for fname, node in zip(filepaths, constructed):
    #         header = f"// {fname}\n"
    #         header += "=" * len(header)
    #         print(header)
    #         print(pformat_ast(node), end="\n\n")

    # if args.json:
    #     print([dump(node, None) for node in constructed])

    return Status.OK


if __name__ == "__main__":
    main()
