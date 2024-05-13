"""Source File Compilation to AST."""

from dataclasses import dataclass
from pathlib import Path

from fhy.lang import ast, from_fhy_source

from ..file_reader import read_file


@dataclass(frozen=True)
class SourceFileAST(object):
    """Data Container mapping File information to ast module build.

    Args:
        path (pathlib.Path): Filepath to Module
        ast (ast.Module): Constructed AST Module

    """

    path: Path
    ast: ast.Module


def build_source_file_ast(source_file_path: Path) -> SourceFileAST:
    """Build an AST Module from a valid Filepath.

    Args:
        source_file_path (Path): Filepath to Module

    Returns:
        SourceFileAST: Dataclass containing both built Module AST and filepath info.

    """
    source_text = read_file(source_file_path)
    source_ast = from_fhy_source(source_text)

    return SourceFileAST(source_file_path, source_ast)
