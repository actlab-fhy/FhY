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

"""Source file compilation to AST."""

import logging
from dataclasses import dataclass
from pathlib import Path

from fhy.lang import ast, from_fhy_source

from ..file_reader import read_file


@dataclass(frozen=True)
class SourceFileAST:
    """Data container mapping file information to ast module build.

    Args:
        path (pathlib.Path): Filepath to module
        ast (ast.Module): Constructed AST module

    """

    path: Path
    ast: ast.Module


def build_source_file_ast(
    source_file_path: Path, log: logging.Logger | None = None
) -> SourceFileAST:
    """Build an AST module from a valid filepath.

    Args:
        source_file_path (Path): Filepath to module
        log (optional, logging.Logger): Inject a logger to control debugging information
            during parsing.

    Returns:
        SourceFileAST: Dataclass containing both built module AST and filepath info.

    Note:
        Provided File path should be validated to exist before use in order
        to provide client with cleaner expected error handling messages.

    """
    source_text = read_file(source_file_path)
    if log is not None:
        source_ast = from_fhy_source(source_text, ast.Source(source_file_path), log)
    else:
        source_ast = from_fhy_source(source_text, ast.Source(source_file_path))

    return SourceFileAST(source_file_path, source_ast)
