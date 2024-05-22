"""Main compilation driver."""

import logging

from fhy import ir

from .ast_program_builder import build_ast_program
from .compilation_options import CompilationOptions
from .workspace import Workspace

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


def compile_fhy(
    workspace: Workspace, options: CompilationOptions, log: logging.Logger = _log
) -> ir.Program:
    """Compile Fhy source into a ir.Program."""
    ast_program = build_ast_program(workspace, options, log)

    return ast_program
