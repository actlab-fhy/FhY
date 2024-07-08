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

"""Container classes to define source positions of an object.

Classes:
    Slice: start and stop positions
    Source: source file or namespace definition
    Span: full context of an object (Source + Slice)

"""

from pathlib import Path


class Slice:
    """General definition of start and stop positions.

    Args:
        start (int): start position
        stop (int): stop position

    """

    start: int
    stop: int

    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Slice)
            and self.start == value.start
            and self.stop == value.stop
        )

    def __repr__(self) -> str:
        return f"{self.start:,d}:{self.stop:,d}"


class Source:
    """Defines source file or namespace.

    Args:
        namespace (str): valid filepath or namespace

    """

    namespace: str | Path

    def __init__(self, namespace: str | Path) -> None:
        self.namespace = namespace

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Source) and self.namespace == value.namespace

    def __repr__(self) -> str:
        return str(self.namespace)


class Span:
    """Context to describe locations of an object.

    Args:
        start_line (int): line start position
        end_line (int): line end position
        start_column (int): column start position
        end_column (int): column end position
        source (Source): source object defining file or namespace

    Attributes:
        source (Source): file or namespace source
        line (_Span): start and stop line positions
        column(_Span):start and stop column positions

    """

    source: Source | None
    line: Slice
    column: Slice

    def __init__(
        self,
        start_line: int,
        end_line: int,
        start_column: int,
        end_column: int,
        source: Source | None = None,
    ) -> None:
        self.source = source
        self.line = Slice(start_line, end_line)
        self.column = Slice(start_column, end_column)

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Span)
            and self.source == value.source
            and self.line == value.line
            and self.column == value.column
        )

    def __repr__(self) -> str:
        text = ""
        if self.source is not None:
            text += f"{self.source} "
        text += f"Lines {self.line} Columns: {self.column}"

        return text
