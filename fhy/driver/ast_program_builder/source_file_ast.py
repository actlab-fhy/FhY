"""Source file compilation to AST."""

from dataclasses import dataclass
from pathlib import Path

from fhy.lang import ast, from_fhy_source

from ..file_reader import read_file


@dataclass(frozen=True)
class SourceFileAST(object):
    """Data container mapping file information to ast module build.

    Args:
        path (pathlib.Path): Filepath to module
        ast (ast.Module): Constructed AST module

    """

    path: Path
    ast: ast.Module


def build_source_file_ast(source_file_path: Path) -> SourceFileAST:
    """Build an AST module from a valid filepath.

    Args:
        source_file_path (Path): Filepath to module

    Returns:
        SourceFileAST: Dataclass containing both built module AST and filepath info.

    Note:
        Provided File path should be validated to exist before use in order
        to provide client with cleaner expected error handling messages.

    """
    source_text = read_file(source_file_path)
    source_ast = from_fhy_source(source_text)

    return SourceFileAST(source_file_path, source_ast)
