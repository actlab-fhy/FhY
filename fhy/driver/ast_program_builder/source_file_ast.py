from dataclasses import dataclass
from pathlib import Path
from fhy.lang import ast
from ..file_reader import read_file
from fhy.lang import from_fhy_source


@dataclass(frozen=True)
class SourceFileAST(object):
    path: Path
    ast: ast.Module


def build_source_file_ast(source_file_path: Path) -> SourceFileAST:
    source_text = read_file(source_file_path)
    source_ast = from_fhy_source(source_text)
    return SourceFileAST(source_file_path, source_ast)
