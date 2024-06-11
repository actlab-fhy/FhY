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

import enum
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeVar, Union

import typer
import typer.core
from typing_extensions import Annotated

from fhy import __version__, ir
from fhy.driver import CompilationOptions, Workspace, compile_fhy
from fhy.driver.file_reader import standard_path
from fhy.lang.ast.pprint import pformat_ast
from fhy.lang.ast.serialization import SerializationOptions
from fhy.lang.ast.serialization.to_json import dump
from fhy.logger import add_file_handler, get_logger

T = TypeVar("T")

app = typer.Typer(
    name="FhY",
    no_args_is_help=True,
    add_completion=False,
)

_cli_log: logging.Logger = get_logger(__name__)

# Make it possible to use environment variable to control help menu display
# NOTE: Typer imports rich module, and on import error sets to None, ignoring typing
#       We will also support not having rich as a dependency of typer
if typer.core.rich is not None:
    typer.core.rich = os.environ.get("FHY_HELPMENU", "rich") or None  # type: ignore


def make_logger(
    verbose: bool = False, file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """Construct a simple logger."""
    level: int = logging.DEBUG if verbose else logging.INFO
    log: logging.Logger = get_logger("FhY", level=level)
    if file is not None:
        add_file_handler(log, file)

    return log


class DefaultPaths:
    """Centralized definitions of path assumptions."""

    hidden_directory: str = ".fhy"
    _log_file: str = "fhy.log"

    @classmethod
    def log_file(cls) -> str:
        return os.path.join(cls.hidden_directory, cls._log_file)


class Status(int, enum.Enum):
    """Exit Codes describing the Status of CLI Request."""

    OK = 0
    FAILED = 1
    INTERRUPTED = 2
    USAGE_ERROR = 3


@dataclass
class CompilationResult:
    """Results of compilation."""

    program: ir.Program
    logger: logging.Logger
    status: Status


def create_hidden_directory(hidden: Path):
    """Create hidden directory."""
    if not hidden.exists():
        os.mkdir(hidden)


def clear_hidden_directory(hidden: Path):
    """Clear hidden directory cache."""
    if hidden.exists():
        shutil.rmtree(hidden)
        _cli_log.info("FhY cache has been cleared")


def _clean_dir(value: bool):
    if not value:
        return
    where: Path = standard_path(os.getcwd())
    hidden: Path = where / DefaultPaths.hidden_directory
    clear_hidden_directory(hidden)
    sys.exit(Status.OK)


def report_version(value: bool):
    """Report version to stdout and exit if true."""
    if value:
        sys.stdout.write(f"FhY v{__version__}\n")
        sys.exit(Status.OK)


def _confirm_arg(value: bool, check: str) -> bool:
    return value or check in sys.argv


def compile_fhy_source(
    main_file: Optional[Path] = None,
    verbose: bool = False,
    log_file: Optional[Path] = None,
    config: Optional[Path] = None,
    force_rebuild: bool = False,
) -> CompilationResult:
    """Parse a source file, convert into high level IR Program."""
    status = Status.OK
    where: Path = standard_path(os.getcwd())
    hidden: Path = where / DefaultPaths.hidden_directory

    # Clear hidden directory if client toggled
    if force_rebuild:
        clear_hidden_directory(hidden)

    # Setup log debugging and hidden directory
    create_hidden_directory(hidden)
    log: logging.Logger = make_logger(verbose, where / DefaultPaths.log_file())
    if log_file is not None:
        add_file_handler(log, log_file, logging.DEBUG if verbose else logging.INFO)

    # NOTE: Inform client we have not currently set this portion up, but scoping
    # here for future backward compatibility.
    if config is not None:
        log.info(
            "Client set configuration file, but this option is NOT yet currently used."
        )

    # Now we start Compiling
    try:
        filepath: Path = standard_path(main_file)

    except (TypeError, FileExistsError) as e:
        log.error(str(e), exc_info=e)
        status = Status.USAGE_ERROR
        sys.exit(status)

    else:
        log.debug(f"Filepath(s) Collected: {filepath}")

    workspace = Workspace(root=filepath)
    options = CompilationOptions(verbose=verbose)

    try:
        program: ir.Program = compile_fhy(workspace, options, log)

    except KeyboardInterrupt as e:
        status = Status.INTERRUPTED
        log.error(
            f"FhY Compilation has been Interrupted by client. Exit {status}.",
            exc_info=e,
        )
        sys.exit(status)

    except Exception as e:
        status = Status.FAILED
        log.error(f"FhY Compilation has Failed. Exit {status}.", exc_info=e)
        sys.exit(status)

    else:
        log.info("FhY compilation completed successfully.")

    return CompilationResult(program=program, logger=log, status=status)


@app.callback(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", help="Show FhY version and exit.", callback=report_version
        ),
    ] = False,
    clean: Annotated[
        bool,
        typer.Option(
            "--clean", help="Clean FhY hidden directory and exit.", callback=_clean_dir
        ),
    ] = False,
):
    """Welcome to FhY!"""
    # NOTE: We check sys.argv to make it possible to place arguments in subcommands
    #       and respond equivalently.
    if _confirm_arg(version, "--version"):
        report_version(True)

    if _confirm_arg(clean, "--clean"):
        _clean_dir(True)


@app.command(
    name="serialize",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def serialize(
    main_file: Annotated[
        Optional[Path],
        typer.Argument(help="Valid filepath to main FhY module source code."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Enable debugging.")
    ] = False,
    log_file: Annotated[
        Optional[Path], typer.Option(help="Provide a filepath to write logs to.")
    ] = None,
    config: Annotated[
        Optional[Path], typer.Option(help="FhY compilation configuration option file.")
    ] = None,
    force_rebuild: Annotated[
        bool, typer.Option("--force-rebuild", help="Clear FhY cache before compiling.")
    ] = False,
    format: Annotated[
        Optional[SerializationOptions],
        typer.Option(
            "--format", "-f", case_sensitive=False, help="Format to serialize to."
        ),
    ] = None,
    indent: Annotated[
        Optional[int],
        typer.Option(
            "--indent",
            "-i",
            case_sensitive=False,
            help="Indentation to apply to serialization for human readability.",
        ),
    ] = None,
):
    """Serialize FhY AST nodes into alternative text representations."""
    compiled: CompilationResult = compile_fhy_source(
        main_file, verbose, log_file, config, force_rebuild
    )

    log: logging.Logger = compiled.logger

    if (program := compiled.program) is None:
        log.error("IR Program not built.")
        sys.exit(Status.FAILED)

    if format is None:
        return compiled.status

    elif format.value == SerializationOptions.JSON:
        sys.stdout.write("\n\n")
        text: str = dump(list(program._components.values()), indent)
        sys.stdout.write(text)
        sys.stdout.write("\n\n")

    elif format.value in (SerializationOptions.PRETTY, SerializationOptions.PRETTYID):
        space: str = (indent or 2) * " "
        show_id: bool = format == SerializationOptions.PRETTYID
        sys.stdout.write("\n\n")
        for key, value in program._components.items():
            text = pformat_ast(value, space, show_id)
            name = key.name_hint
            sys.stdout.write(f"\\\\ {name}\n{'=' * (len(name) + 3)}\n")
            sys.stdout.write(text)
            sys.stdout.write("\n\n")

    else:
        log.error(f"Unsupported or Invalid Serialization Format: {format.value}")
        compiled.status = Status.USAGE_ERROR

    return compiled.status
