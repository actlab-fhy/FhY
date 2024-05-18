"""Main Compilation Driver."""

from fhy import ir

from .ast_program_builder import build_ast_program
from .compilation_options import CompilationOptions
from .workspace import Workspace


def compile_fhy(workspace: Workspace, options: CompilationOptions) -> ir.Program:
    """Compile Fhy source into a Program."""
    ast_program = build_ast_program(workspace, options)

    return ast_program
