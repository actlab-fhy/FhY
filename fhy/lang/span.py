# TODO Jason: Add docstring
from typing import Optional


class _Span:
    start: int
    end: int

    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop

    def __repr__(self) -> str:
        return f"{self.start:,d}:{self.stop:,d}"


# TODO: Jason: Create Source object that can track the source file
class Source(object):
    namespace: str

    def __init__(self, namespace) -> None:
        self.namespace = namespace

    def __repr__(self) -> str:
        return self.namespace


class Span(object):
    # TODO Jason: Add docstring
    source: Source
    line: _Span
    column: _Span

    def __init__(
        self,
        start_line: int,
        end_line: int,
        start_column: int,
        end_column: int,
        source: Optional[Source] = None,
    ) -> None:
        self.source = source
        self.line = _Span(start_line, end_line)
        self.column = _Span(start_column, end_column)

    def __repr__(self) -> str:
        text = ""
        if self.source is not None:
            text += f"{self.source} "
        text += f"Lines {self.line} Columns: {self.column}"
        return text

    # TODO Jason: Implement the functionality of this class
